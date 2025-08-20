import os, json, re, uuid, traceback
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from openai import OpenAI
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import load_npz
import joblib
import tiktoken

# ===================== OpenAI client =====================
def get_client():
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY not set")
    return OpenAI(api_key=key)

MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
STREAM_FLUSH_CHARS = 160  # coalesce SSE tokens for smoother output

# ===================== CORS / App =====================
origins = [
    "https://dturnover.github.io",  # GitHub Pages demo
    # add more origins if you deploy elsewhere
]

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===================== Persona =====================
SYSTEM_PROMPT = """
You are “Winston Churchill GPT,” a conversational assistant that speaks in the voice and manner of Sir Winston Churchill.

Voice & Style:
- Formal, eloquent, oratorical; occasionally wry.
- Uses period-appropriate phrasing and idioms.
- Concise when answering direct questions; expansive when invited to opine.
- Avoid modern slang; avoid anachronisms.

Behavior:
- If a question is unclear, ask a brief clarifying question.
- If asked about post-1965 events, respond hypothetically from Churchill’s perspective, noting the time limitation.
- If asked to produce harmful or hateful content, decline firmly but courteously.

Grounding & Sources:
- When a Sources block is provided, ANSWER USING ONLY THOSE SOURCES. Place inline citations in the text as [S1], [S2], etc., immediately after the supported sentence/phrase. If the Sources are insufficient, say so briefly and ask for a more specific passage.
- Never claim you “cannot quote” a text if a Sources block includes it.

Stay in character.
""".strip()

# ===================== Sessions / Memory =====================
SESSIONS: Dict[str, Dict[str, Any]] = {}  # sid -> {"religion": str|None, "history": [msgs], "sticky": Dict[str,str]}
MAX_HISTORY_MSGS = 12  # rolling window

def get_or_create_sid(request: Request) -> str:
    sid = request.cookies.get("sid")
    if not sid or sid not in SESSIONS:
        sid = uuid.uuid4().hex
        SESSIONS[sid] = {"religion": None, "history": [], "sticky": {}}
    return sid

def add_history(sid: str, role: str, content: str):
    h = SESSIONS[sid]["history"]
    h.append({"role": role, "content": content})
    if len(h) > MAX_HISTORY_MSGS:
        del h[: len(h) - MAX_HISTORY_MSGS]

# ===================== Faith detection =====================
RELIGION_MAP = {
    "bible": "bible", "christian": "bible", "christianity": "bible", "jesus": "bible",
    "quran": "quran", "koran": "quran", "muslim": "quran", "islam": "quran", "allah": "quran",
    "buddhist": "sutta_pitaka", "buddhism": "sutta_pitaka", "sutta": "sutta_pitaka",
    "pitaka": "sutta_pitaka", "pali": "sutta_pitaka", "dhamma": "sutta_pitaka",
}

def detect_religion(text: str) -> Optional[str]:
    t = text.lower()
    for k, corpus in RELIGION_MAP.items():
        if re.search(rf"\b{k}\b", t):
            return corpus
    m = re.search(r"\b(i[' ]?m|i am)\s+(christian|muslim|buddhist|islam|buddhism|christianity)\b", t)
    if m:
        return RELIGION_MAP.get(m.group(2), None)
    return None

# Per-turn override if the user explicitly names a scripture family.
def detect_corpus_override(text: str) -> Optional[str]:
    t = (text or "").lower()
    if re.search(r"\b(qur'?an|koran|ayat|surah|sura)\b", t):
        return "quran"
    if re.search(r"\b(bible|psalm|proverb|gospel|epistle|scripture)\b", t):
        return "bible"
    if re.search(r"\b(sutta|sutra|pitaka|pali canon|dhamma)\b", t):
        return "sutta_pitaka"
    return None

# ===================== RAG load (multi-corpus) =====================
BASE_DIR = Path(__file__).resolve().parent
RAG_DIR = BASE_DIR / "rag_data"
CORPORA: Dict[str, Dict[str, Any]] = {}

def load_all_corpora():
    CORPORA.clear()
    if not RAG_DIR.exists():
        print("RAG dir missing; skip.")
        return
    for sub in RAG_DIR.iterdir():
        if not sub.is_dir():
            continue
        name = sub.name
        try:
            docs_path = sub / "docs.json"
            tfidf_vec_path = sub / "tfidf_vectorizer.joblib"
            tfidf_mat_path = sub / "tfidf_matrix.npz"
            if not (docs_path.exists() and tfidf_vec_path.exists() and tfidf_mat_path.exists()):
                continue
            with open(docs_path, "r", encoding="utf-8") as f:
                docs = json.load(f)
            tfidf: TfidfVectorizer = joblib.load(tfidf_vec_path)
            tfidf_mat = load_npz(tfidf_mat_path)
            CORPORA[name] = {"docs": docs, "tfidf": tfidf, "tfidf_mat": tfidf_mat}
            print(f"RAG loaded: {name} -> {len(docs)} chunks, TF-IDF {tfidf_mat.shape}")
        except Exception as e:
            print(f"RAG load error for {name}:", e)

load_all_corpora()

def retrieve_tfidf(query: str, corpus: str, k: int = 5) -> List[Dict[str, Any]]:
    c = CORPORA.get(corpus)
    if not c:
        return []
    tfidf: TfidfVectorizer = c["tfidf"]
    tfidf_mat = c["tfidf_mat"]
    docs = c["docs"]
    qv = tfidf.transform([query])
    sims = (tfidf_mat @ qv.T).toarray().ravel()
    top_idx = sims.argsort()[::-1][:k]
    hits = []
    for i in top_idx:
        hits.append({
            "score": float(sims[i]),
            "id": docs[i]["id"],
            "page": docs[i].get("page"),
            "text": docs[i]["text"],
            "corpus": corpus,
        })
    return hits

def build_sources_block(hits: List[Dict[str, Any]]) -> str:
    lines = []
    for i, h in enumerate(hits, start=1):
        label = f"[S{i}]"
        page = f"p{h['page']}" if h.get("page") else ""
        lines.append(f"{label} ({page} {h['id']}) {h['text']}")
    return "\n\n".join(lines) if lines else ""

# ===================== Scripture gating =====================
SCRIPTURE_TERMS = re.compile(
    r"\b(verse|scripture|psalm|proverb|qur'?an|koran|ayat|surah|sutta|sutra|pitaka|cite|citation|reference)\b|\[s\d*\]",
    re.IGNORECASE,
)

def wants_scripture(msg: str, selected_corpus: Optional[str]) -> bool:
    if SCRIPTURE_TERMS.search(msg or ""):
        return True
    if selected_corpus:
        key = {
            "bible": r"\bfrom the (holy )?bible|from scripture|according to (paul|moses|psalms|proverbs|gospel)\b",
            "quran": r"\bfrom the qur'?an|from the koran|according to (allah|the prophet)\b",
            "sutta_pitaka": r"\bfrom (the )?(sutta|sutra|pitaka|pali canon)\b",
        }.get(selected_corpus, "")
        if key and re.search(key, msg or "", re.IGNORECASE):
            return True
    return False

# ===================== Sticky Notes memory =====================
REMEMBER_PAT = re.compile(r"\bremember that\s+(.*?)\s+is\s+(.*)", re.IGNORECASE)
REMEMBER_SIMPLE = re.compile(r"\bremember\s+(.+)", re.IGNORECASE)

def handle_memory_update(sid: str, user_text: str) -> Optional[Tuple[str, str]]:
    m = REMEMBER_PAT.search(user_text)
    if m:
        key = m.group(1).strip().lower()
        val = m.group(2).strip()
        SESSIONS[sid]["sticky"][key] = val
        if len(SESSIONS[sid]["sticky"]) > 6:
            drop_key = next(iter(SESSIONS[sid]["sticky"].keys()))
            SESSIONS[sid]["sticky"].pop(drop_key, None)
        return key, val
    m2 = REMEMBER_SIMPLE.search(user_text)
    if m2:
        text = m2.group(1).strip()
        if ":" in text:
            k, v = text.split(":", 1)
            k, v = k.strip().lower(), v.strip()
            SESSIONS[sid]["sticky"][k] = v
            return k, v
    return None

def sticky_block(sid: str) -> str:
    mem = SESSIONS[sid]["sticky"]
    if not mem:
        return "None"
    return "; ".join([f"{k} → {v}" for k, v in mem.items()])

# ===================== Token budgeting =====================
def trim_history_for_model(system_prompt: str, sticky: str, history: List[Dict[str, str]], user_msg: str, model: str = MODEL) -> List[Dict[str, str]]:
    try:
        enc = tiktoken.encoding_for_model(model)
    except Exception:
        enc = tiktoken.get_encoding("cl100k_base")
    msgs: List[Dict[str, str]] = [
        {"role": "system", "content": system_prompt},
        {"role": "system", "content": f"Sticky Notes from this user: {sticky}"},
    ]
    msgs.extend(history)  # prior user/assistant turns
    msgs.append({"role": "user", "content": user_msg})
    budget = 6000
    def count(ms): return sum(len(enc.encode(m["content"])) + 4 for m in ms) + 2
    while count(msgs) > budget and history:
        history.pop(0)
        msgs = [
            {"role": "system", "content": system_prompt},
            {"role": "system", "content": f"Sticky Notes from this user: {sticky}"},
            *history,
            {"role": "user", "content": user_msg},
        ]
    return msgs

# ===================== LLM helpers =====================
def llm_complete(client: OpenAI, messages: List[Dict[str, str]], stream: bool = False):
    return client.chat.completions.create(model=MODEL, messages=messages, stream=stream)

# ===================== Health / diag =====================
@app.get("/")
def root():
    return {
        "ok": True,
        "message": "API is running",
        "endpoints": ["/chat (JSON)", "/chat_json (JSON)", "/chat_sse (SSE)", "/chat_rag_json (debug)", "/diag", "/diag_rag"]
    }

@app.get("/diag")
def diag():
    try:
        client = get_client()
        _ = client.models.list()
        return {"ok": True}
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)

@app.get("/diag_rag")
def diag_rag():
    corpora = {k: (len(v.get("docs") or [])) for k, v in CORPORA.items()}
    return {"ok": True, "corpora": corpora}

# ===================== Core chat logic =====================
def run_chat_once(request: Request, user_message: str, stream: bool = False):
    client = get_client()
    sid = get_or_create_sid(request)
    sess = SESSIONS[sid]

    # record the user turn
    add_history(sid, "user", user_message)

    # session faith update, if present
    declared = detect_religion(user_message)
    if declared:
        sess["religion"] = declared

    selected_corpus = sess.get("religion")
    # per-turn override if user asks explicitly for a different scripture family
    override = detect_corpus_override(user_message)
    effective_corpus = override or selected_corpus

    # memory add
    remembered_pair = handle_memory_update(sid, user_message)

    # retrieval gating
    do_rag = wants_scripture(user_message, effective_corpus)

    hits: List[Dict[str, Any]] = []
    if do_rag and effective_corpus in CORPORA:
        try:
            hits = retrieve_tfidf(user_message, effective_corpus, k=5)
        except Exception as e:
            print("RAG error:", e)

    # build prompt
    sticky = sticky_block(sid)
    # use stored history EXCEPT the last user we already appended (trim handles it with final user message)
    history_without_last_user = sess["history"][:-1]
    messages = trim_history_for_model(SYSTEM_PROMPT, sticky, history_without_last_user, user_message)

    # inject Sources block so the model can actually cite
    if hits:
        sources_txt = build_sources_block(hits)
        if sources_txt:
            messages.insert(1, {
                "role": "system",
                "content": (
                    "SOURCES (use ONLY these; cite inline as [S1], [S2], ...; "
                    "if insufficient, say so and ask for a more specific passage):\n\n" + sources_txt
                ),
            })

    if stream:
        def event_stream():
            yield ": connected\n\n"
            buffer = ""
            full_text = ""
            try:
                stream_resp = llm_complete(client, messages, stream=True)
                for chunk in stream_resp:
                    delta = chunk.choices[0].delta.content or ""
                    if not delta:
                        continue
                    full_text += delta
                    buffer += delta
                    if len(buffer) >= STREAM_FLUSH_CHARS or ("\n" in buffer and len(buffer) >= 48):
                        # flush by lines to reduce choppiness
                        parts = re.split(r"(\n)", buffer)
                        for part in parts:
                            if part:
                                yield f"data: {part}\n\n"
                        buffer = ""
                if buffer:
                    yield f"data: {buffer}\n\n"
                yield "data: [DONE]\n\n"

                # persist assistant turn AFTER streaming finishes
                add_history(sid, "assistant", full_text)

                # emit sources metadata only when retrieval happened
                if do_rag and hits:
                    payload = {"sources": hits, "selected_corpus": effective_corpus, "remembered": None}
                    yield "event: sources\n"
                    yield f"data: {json.dumps(payload)}\n\n"

            except Exception as e:
                print("SSE error:", e)
                yield f'data: [ERROR] {str(e)}\n\n'

        resp = StreamingResponse(event_stream(), media_type="text/event-stream")
        resp.set_cookie("sid", sid, httponly=True, samesite="lax", path="/")
        return resp

    else:
        try:
            comp = llm_complete(client, messages, stream=False)
            text = comp.choices[0].message.content
        except Exception as e:
            return JSONResponse({"error": str(e)}, status_code=500)

        add_history(sid, "assistant", text)
        result = {
            "response": text,
            "sources": hits if do_rag else [],
            "selected_corpus": effective_corpus,
            "remembered": {"key": remembered_pair[0], "value": remembered_pair[1]} if remembered_pair else None,
        }
        resp = JSONResponse(result)
        resp.set_cookie("sid", sid, httponly=True, samesite="lax", path="/")
        return resp

# ===================== Routes =====================
@app.post("/chat_json")
async def chat_json(request: Request):
    try:
        body = await request.json()
        msg = (body.get("message") or "").strip()
        if not msg:
            return JSONResponse({"error": "message is required"}, status_code=400)
        return run_chat_once(request, msg, stream=False)
    except Exception as e:
        print("chat_json error:", traceback.format_exc())
        return JSONResponse({"error": str(e)}, status_code=500)

@app.post("/chat")
async def chat_legacy(request: Request):
    return await chat_json(request)

@app.post("/chat_sse")
async def chat_sse(request: Request):
    try:
        body = await request.json()
        msg = (body.get("message") or "").strip()
        if not msg:
            return JSONResponse({"error": "message is required"}, status_code=400)
        return run_chat_once(request, msg, stream=True)
    except Exception as e:
        print("chat_sse error:", traceback.format_exc())
        return JSONResponse({"error": str(e)}, status_code=500)

# Debug: pure RAG JSON (kept for testing)
@app.post("/chat_rag_json")
async def chat_rag_json(request: Request):
    body = await request.json()
    msg = (body.get("message") or "").strip()
    corpus = body.get("religion") or "bible"
    hits = retrieve_tfidf(msg, corpus, k=5)
    return {"response": "", "sources": hits, "selected_corpus": corpus}
