import os, json, re, uuid, traceback
from pathlib import Path
from typing import Dict, Any, List, Optional, Iterable

import numpy as np
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse, Response, PlainTextResponse
from openai import OpenAI
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import load_npz
import joblib

# ==============================
# OpenAI client
# ==============================
def get_client() -> OpenAI:
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY not set")
    return OpenAI(api_key=key)

MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")  # override in env if needed

# ==============================
# CORS (add your frontend origins)
# ==============================
origins = [
    "https://dturnover.github.io",    # GitHub Pages demo
    # "https://<your-domain>",        # add when ready
]
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==============================
# Persona (Churchill voice + scripture rules)
# ==============================
BASE_SYSTEM_PROMPT = """
You are “Winston Churchill GPT,” a calm, encouraging spiritual guide for fighters.
Voice & Style:
- Formal, eloquent, oratorical; occasionally wry; concise when needed.
- Avoid modern slang and anachronisms.

Behavior:
- Be helpful and pastoral. If the session has a known faith (see Session Meta), weave in a SHORT, relevant verse (1–2 lines) when it helps, and especially when asked for “verse/scripture/quote”.
- If no faith is known and the user asks for scripture, briefly ask which tradition (e.g., Bible, Qur’an, Suttas).
- Never invent citations. Only cite passages we provide in the “Sources” block as [S1], [S2], etc. If no sources are provided, do NOT mention sources or citations.
- If a question is unclear, ask a brief clarifying question.

When quoting scripture:
- Quote verbatim from the provided Sources block and cite as [S#].
- Keep the verse short unless the user asks for more.
""".strip()

def build_dynamic_system(religion: Optional[str], have_sources: bool) -> str:
    parts = [BASE_SYSTEM_PROMPT, ""]
    parts.append("Session Meta:")
    parts.append(f"- Faith: {religion or 'unknown'}")
    if have_sources:
        parts.append("- Sources provided below. Use only those passages when quoting and cite [S#].")
    else:
        parts.append("- No sources provided for this turn; do not mention citations.")
    return "\n".join(parts)

# ==============================
# In-memory session store
# ==============================
SESSIONS: Dict[str, Dict[str, Any]] = {}  # sid -> {"religion": str|None, "history": list[dict]}
MAX_TURNS = 24   # cap history so token use stays modest

def get_or_create_sid(request: Request) -> str:
    sid = request.cookies.get("sid")
    if (not sid) or (sid not in SESSIONS):
        sid = uuid.uuid4().hex
        SESSIONS[sid] = {"religion": None, "history": []}
    return sid

def remember(sid: str, role: str, content: str) -> None:
    session = SESSIONS.setdefault(sid, {"religion": None, "history": []})
    session["history"].append({"role": role, "content": content})
    if len(session["history"]) > MAX_TURNS:
        session["history"] = session["history"][-MAX_TURNS:]

# ==============================
# Religion detection
# ==============================
RELIGION_MAP = {
    "bible": "bible", "christian": "bible", "christianity": "bible", "jesus": "bible",
    "quran": "quran", "koran": "quran", "muslim": "quran", "islam": "quran", "allah": "quran",
    "buddhist": "sutta_pitaka", "buddhism": "sutta_pitaka", "sutta": "sutta_pitaka",
    "pitaka": "sutta_pitaka", "pali": "sutta_pitaka", "dhamma": "sutta_pitaka",
}

def detect_religion(text: str) -> Optional[str]:
    t = text.lower()

    # Direct keywords
    for k, corpus in RELIGION_MAP.items():
        if re.search(rf"\b{k}\b", t):
            return corpus

    # "I am / I'm <faith>"
    m = re.search(
        r"\b(i[' ]?m|i am)\s+(christian|muslim|buddhist|islam|buddhism|christianity)\b",
        t
    )
    if m:
        return RELIGION_MAP.get(m.group(2), None)
    return None

# ==============================
# RAG load (multi-corpus)
# ==============================
BASE_DIR = Path(__file__).resolve().parent
RAG_DIR = BASE_DIR / "rag_data"
CORPORA: Dict[str, Dict[str, Any]] = {}  # name -> {docs, tfidf, tfidf_mat}

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
        # Keep raw text; the model will quote a short line
        lines.append(f"{label} ({page} {h['id']}) {h['text']}")
    return "\n\n".join(lines) if lines else ""

# ==============================
# Heuristic: should we attach RAG?
# ==============================
SCRIPTURE_KEYWORDS = [
    "verse", "scripture", "passage", "quote", "psalm", "proverb",
    "gospel", "surah", "ayah", "hadith", "sutta", "sutra", "bible", "quran"
]
PASTORAL_KEYWORDS = [
    "strength","courage","fear","anxiety","hope","grief","patience","perseverance",
    "determination","temptation","forgiveness","love","faith","doubt","guidance",
    "struggle","suffering","trial","hardship","discipline","fight","battle","healing","peace"
]

def should_attach_rag(user_text: str, religion: Optional[str]) -> bool:
    if not religion:
        # Only attach by explicit request if faith unknown
        return any(w in user_text.lower() for w in SCRIPTURE_KEYWORDS)
    t = user_text.lower()
    # If faith is known: attach when asked for scripture OR when pastoral themes appear
    if any(w in t for w in SCRIPTURE_KEYWORDS):
        return True
    if any(w in t for w in PASTORAL_KEYWORDS):
        return True
    # Otherwise, skip for small talk/meta
    return False

# ==============================
# Helpers to call OpenAI (streaming)
# ==============================
def build_messages(sid: str, user_text: str, hits: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    religion = SESSIONS[sid]["religion"]
    have_sources = len(hits) > 0
    system = build_dynamic_system(religion, have_sources)

    msgs: List[Dict[str, str]] = []
    msgs.append({"role": "system", "content": system})

    if have_sources:
        # Provide sources as a separate system message
        src_block = build_sources_block(hits)
        msgs.append({
            "role": "system",
            "content": "Sources:\n" + src_block
        })

    # Short memory
    for m in SESSIONS[sid]["history"]:
        msgs.append(m)

    msgs.append({"role": "user", "content": user_text})
    return msgs

def stream_chat_completion(messages: List[Dict[str, str]]) -> Iterable[str]:
    client = get_client()
    # Use Chat Completions for wide compatibility
    resp = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=0.7,
        stream=True,
    )
    for chunk in resp:
        if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content

# ==============================
# Health / diag
# ==============================
@app.get("/")
def root():
    return {
        "ok": True,
        "message": "API is running",
        "endpoints": ["/chat (stream)", "/chat_sse", "/chat_json", "/diag", "/diag_rag", "/chat_rag_json"]
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

# ==============================
# Streaming chat (plain text chunks)
# ==============================
@app.post("/chat")
async def chat(request: Request):
    try:
        body = await request.json()
        user_text: str = body.get("message", "").strip()
        if not user_text:
            return JSONResponse({"error": "Missing 'message'."}, status_code=400)

        sid = get_or_create_sid(request)
        # detect religion (sticky)
        maybe = detect_religion(user_text)
        if maybe:
            SESSIONS[sid]["religion"] = maybe

        religion = SESSIONS[sid]["religion"]
        hits: List[Dict[str, Any]] = []
        if should_attach_rag(user_text, religion):
            hits = retrieve_tfidf(user_text, religion or "bible", k=5)

        messages = build_messages(sid, user_text, hits)

        def gen():
            full = []
            try:
                for token in stream_chat_completion(messages):
                    full.append(token)
                    yield token
            except Exception as e:
                yield f"\n[Error streaming: {e}]\n"
            finally:
                # remember turn
                text = "".join(full).strip()
                remember(sid, "user", user_text)
                if text:
                    remember(sid, "assistant", text)

        resp = StreamingResponse(gen(), media_type="text/plain; charset=utf-8")
        resp.set_cookie("sid", sid, httponly=True, samesite="lax")
        return resp
    except Exception as e:
        traceback.print_exc()
        return JSONResponse({"error": str(e)}, status_code=500)

# ==============================
# Streaming chat via Server-Sent Events (SSE)
# ==============================
@app.post("/chat_sse")
async def chat_sse(request: Request):
    try:
        body = await request.json()
        user_text: str = body.get("message", "").strip()
        if not user_text:
            return PlainTextResponse("", status_code=400)

        sid = get_or_create_sid(request)
        maybe = detect_religion(user_text)
        if maybe:
            SESSIONS[sid]["religion"] = maybe

        religion = SESSIONS[sid]["religion"]
        hits: List[Dict[str, Any]] = []
        if should_attach_rag(user_text, religion):
            hits = retrieve_tfidf(user_text, religion or "bible", k=5)

        messages = build_messages(sid, user_text, hits)

        def sse_gen():
            yield ": connected\n\n"
            full = []
            try:
                for token in stream_chat_completion(messages):
                    full.append(token)
                    # basic SSE data frame
                    yield f"data: {token}\n\n"
            except Exception as e:
                yield f"event: error\ndata: {json.dumps({'error': str(e)})}\n\n"
            finally:
                text = "".join(full).strip()
                remember(sid, "user", user_text)
                if text:
                    remember(sid, "assistant", text)
                # emit sources event if any
                if hits:
                    src = {"sources": hits, "selected_corpus": religion}
                    yield f"event: sources\ndata: {json.dumps(src)}\n\n"
                yield "data: [DONE]\n\n"

        resp = StreamingResponse(sse_gen(), media_type="text/event-stream; charset=utf-8")
        resp.set_cookie("sid", sid, httponly=True, samesite="lax")
        return resp
    except Exception as e:
        traceback.print_exc()
        return PlainTextResponse("", status_code=500)

# ==============================
# Non-stream JSON (with sources)
# ==============================
@app.post("/chat_json")
async def chat_json(request: Request):
    try:
        body = await request.json()
        user_text: str = body.get("message", "").strip()
        if not user_text:
            return JSONResponse({"error": "Missing 'message'."}, status_code=400)

        sid = get_or_create_sid(request)
        maybe = detect_religion(user_text)
        if maybe:
            SESSIONS[sid]["religion"] = maybe

        religion = SESSIONS[sid]["religion"]
        hits: List[Dict[str, Any]] = []
        if should_attach_rag(user_text, religion):
            hits = retrieve_tfidf(user_text, religion or "bible", k=5)

        messages = build_messages(sid, user_text, hits)

        # one-shot (non-stream)
        client = get_client()
        comp = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=0.7,
        )
        text = comp.choices[0].message.content.strip()

        remember(sid, "user", user_text)
        remember(sid, "assistant", text)

        return {
            "response": text,
            "sources": hits,
            "selected_corpus": religion,
        }
    except Exception as e:
        traceback.print_exc()
        return JSONResponse({"error": str(e)}, status_code=500)

# ==============================
# Direct RAG test (existing)
# ==============================
@app.post("/chat_rag_json")
async def chat_rag_json(request: Request):
    """
    Bypasses heuristics: you can specify `religion` explicitly to test retrieval + answer.
    """
    try:
        body = await request.json()
        user_text: str = body.get("message", "").strip()
        chosen: Optional[str] = body.get("religion")
        if not user_text:
            return JSONResponse({"error": "Missing 'message'."}, status_code=400)
        if not chosen:
            return JSONResponse({"error": "Missing 'religion'."}, status_code=400)

        hits = retrieve_tfidf(user_text, chosen, k=5)
        messages = build_messages("test-"+uuid.uuid4().hex, user_text, hits)

        client = get_client()
        comp = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=0.7,
        )
        text = comp.choices[0].message.content.strip()
        return {"response": text, "sources": hits, "selected_corpus": chosen}
    except Exception as e:
        traceback.print_exc()
        return JSONResponse({"error": str(e)}, status_code=500)
