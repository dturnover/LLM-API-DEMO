import os, json, re, uuid, traceback, time
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
STREAM_FLUSH_CHARS = 120  # coalesce SSE tokens ~120 chars

# ===================== CORS / App =====================
origins = [
    "https://dturnover.github.io",  # your demo site
    # add your prod site(s) here as needed
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

When sources are provided, ground your answer ONLY in those sources, citing inline as [S1], [S2], etc. If insufficient, say so and ask for a more specific passage.
Stay in character.
""".strip()

# ===================== Sessions / Memory =====================
SESSIONS: Dict[str, Dict[str, Any]] = {}  # sid -> {"religion": str|None, "history": [msgs], "sticky": Dict[str,str]}
MAX_HISTORY_MSGS = 12  # rolling window (user+assistant messages)

def get_or_create_sid(request: Request) -> str:
    sid = request.cookies.get("sid")
    if not sid or sid not in SESSIONS:
        sid = uuid.uuid4().hex
        SESSIONS[sid] = {"religion": None, "history": [], "sticky": {}}
    return sid

def session(request: Request) -> Dict[str, Any]:
    sid = get_or_create_sid(request)
    return SESSIONS[sid]

def add_history(sid: str, role: str, content: str):
    h = SESSIONS[sid]["history"]
    h.append({"role": role, "content": content})
    # trim to last N messages
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

def is_set_faith_message(text: str) -> bool:
    return detect_religion(text) is not None

# ===================== RAG load (multi-corpus) =====================
BASE_DIR = Path(__file__).resolve().parent
RAG_DIR = BASE_DIR / "rag_data"

# CORPORA[name] = {"docs": List[dict], "tfidf": TfidfVectorizer, "tfidf_mat": csr_matrix}
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
    qv = tfidf.transform([query])                # (1, V)
    sims = (tfidf_mat @ qv.T).toarray().ravel()  # simple similarity proxy
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
    return "\n\n".join(lines) if lines else "No sources found."

# ===================== Scripture gating =====================
SCRIPTURE_TERMS = re.compile(
    r"\b(verse|scripture|psalm|proverb|qur'?an|koran|ayat|hadith|sutta|sutra|cite|citation|reference)\b|\[s\d*\]",
    re.IGNORECASE,
)

def wants_scripture(msg: str, selected_corpus: Optional[str]) -> bool:
    if SCRIPTURE_TERMS.search(msg or ""):
        return True
    # Also trigger if user says "from the <corpus>"
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
    """Return (key, value) if a memory was added."""
    m = REMEMBER_PAT.search(user_text)
    if m:
        key = m.group(1).strip().lower()
        val = m.group(2).strip()
        SESSIONS[sid]["sticky"][key] = val
        # keep at most 6
        if len(SESSIONS[sid]["sticky"]) > 6:
            # drop oldest
            drop_key = next(iter(SESSIONS[sid]["sticky"].keys()))
            SESSIONS[sid]["sticky"].pop(drop_key, None)
        return key, val
    m2 = REMEMBER_SIMPLE.search(user_text)
    if m2:
        text = m2.group(1).strip()
        # naive split "X: Y" or "X = Y"
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

# ===================== Token budgeting (trim history) =====================
def trim_history_for_model(system_prompt: str, sticky: str, history: List[Dict[str, str]], user_msg: str, model: str = MODEL) -> List[Dict[str, str]]:
    enc = tiktoken.encoding_for_model("gpt-4o-mini") if "gpt-4" in model else tiktoken.get_encoding("cl100k_base")
    # Build candidate messages
    msgs: List[Dict[str, str]] = [
        {"role": "system", "content": system_prompt},
        {"role": "system", "content": f"Sticky Notes from this user: {sticky}"},
    ]
    # add rolling history
    for m in history:
        msgs.append(m)
    # final user
    msgs.append({"role": "user", "content": user_msg})

    # rough token count; budget ~6k (gpt-4o-mini supports much more, but keep safe)
    budget = 6000
    def count(ms):
        return sum(len(enc.encode(m["content"])) + 4 for m in ms) + 2

    while count(msgs) > budget and len(history) > 0:
        # drop oldest pair from history (after the system+sticky)
        # remove the third element (index 2) if it's user/assistant alternating
        for i in range(2, len(msgs)):
            if msgs[i]["role"] in ("user", "assistant"):
                del msgs[i]
                break
        # also drop from stored history accordingly
        if SESSIONS:
            # best-effort shrink: pop from the front
            if SESSIONS and history:
                history.pop(0)
    return msgs

# ===================== LLM call helpers =====================
def llm_complete(client: OpenAI, messages: List[Dict[str, str]], stream: bool = False):
    if stream:
        return client.chat.completions.create(model=MODEL, messages=messages, stream=True)
    else:
        return client.chat.completions.create(model=MODEL, messages=messages, stream=False)

def build_answer_with_sources(text: str, hits: List[Dict[str, Any]]) -> str:
    if not hits:
        return text
    # Append inline citations [S1], [S2] only if user asked; the model is instructed to cite when sources are provided.
    # We just ensure sources are available; the model will place [S#] where appropriate.
    return text

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
    add_history(sid, "user", user_message)

    # 1) Set faith if the user declares it — short ack, no RAG
    selected_corpus = sess.get("religion")
    declared = detect_religion(user_message)
    if declared:
        sess["religion"] = declared
        selected_corpus = declared

    # 2) Sticky Notes update if any
    remembered_pair = handle_memory_update(sid, user_message)

    # 3) Scripture gating
    do_rag = wants_scripture(user_message, selected_corpus)

    hits: List[Dict[str, Any]] = []
    if do_rag and selected_corpus in CORPORA:
        try:
            hits = retrieve_tfidf(user_message, selected_corpus, k=5)
        except Exception as e:
            print("RAG error:", e)

    # 4) Build messages with system, sticky, history, and user
    sticky = sticky_block(sid)
    messages = trim_history_for_model(SYSTEM_PROMPT, sticky, sess["history"][:-1], user_message)  # history minus final user we already added above
    # Re-add the last user message (trim_history built with it, but our stored history might have older state)
    # Ensure last message is the user's current prompt (already ensured in trim_history)

    # 5) Call LLM (stream or not)
    if stream:
        def event_stream():
            # --- SSE: send initial "connected" ping
            yield ": connected\n\n"
            buffer = ""

            try:
                stream_resp = llm_complete(client, messages, stream=True)
                for chunk in stream_resp:
                    delta = chunk.choices[0].delta.content or ""
                    if not delta:
                        continue
                    buffer += delta
                    if len(buffer) >= STREAM_FLUSH_CHARS or ("\n" in buffer and len(buffer) >= 40):
                        # flush
                        for part in re.split(r"(\n)", buffer):
                            if part == "":
                                continue
                            yield f"data: {part}\n\n"
                        buffer = ""
                # flush remaining
                if buffer:
                    yield f"data: {buffer}\n\n"

                # Close the text stream
                yield "data: [DONE]\n\n"

                # Sources event ONLY if we actually did retrieval
                if do_rag and hits:
                    payload = {"sources": hits, "selected_corpus": selected_corpus, "remembered": None}
                    yield "event: sources\n"
                    yield f"data: {json.dumps(payload)}\n\n"

            except Exception as e:
                print("SSE error:", e)
                yield f'data: [ERROR] {str(e)}\n\n'

        resp = StreamingResponse(event_stream(), media_type="text/event-stream")
        # ensure cookie
        resp.set_cookie("sid", sid, httponly=True, samesite="lax", path="/")
        return resp

    else:
        try:
            comp = llm_complete(client, messages, stream=False)
            text = comp.choices[0].message.content
        except Exception as e:
            return JSONResponse({"error": str(e)}, status_code=500)

        # store assistant history
        add_history(sid, "assistant", text)

        result = {
            "response": text,
            "sources": hits if do_rag else [],
            "selected_corpus": selected_corpus,
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

# Back-compat for frontends expecting /chat (JSON)
@app.post("/chat")
async def chat_legacy(request: Request):
    return await chat_json(request)

# SSE streaming endpoint
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
