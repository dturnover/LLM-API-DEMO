import os, json, re, uuid, time, traceback
from pathlib import Path
from typing import Dict, Any, List, Optional

import numpy as np  # noqa: F401  (kept for future use)
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from openai import OpenAI
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import load_npz
import joblib

# ========= OpenAI client =========
def get_client():
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY not set")
    return OpenAI(api_key=key)

# ========= CORS =========
origins = [
    "https://dturnover.github.io",  # GitHub Pages demo
    # "https://<your-wp-domain>",   # add when ready
]
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ========= Persona =========
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

Signature touches (use sparingly):
- “Keep buggering on.” / “We shall never surrender.”
- Metaphors of storms, resolve, destiny.

When sources are provided, ground your answer ONLY in those sources, citing inline as [S1], [S2], etc. If insufficient, say so and ask for a more specific passage.
Stay in character throughout.
""".strip()

# ========= In-memory session (per instance) =========
SESSIONS: Dict[str, Dict[str, Any]] = {}  # sid -> {"religion": "bible"/"quran"/"sutta_pitaka"}

def get_or_create_sid(request: Request) -> str:
    sid = request.cookies.get("sid")
    if not sid or sid not in SESSIONS:
        sid = uuid.uuid4().hex
        SESSIONS[sid] = {"religion": None}
    return sid

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

# ========= RAG load (multi-corpus) =========
BASE_DIR = Path(__file__).resolve().parent
RAG_DIR = BASE_DIR / "rag_data"

# corpora[name] = {
#   "docs": List[dict],
#   "tfidf": TfidfVectorizer,
#   "tfidf_mat": csr_matrix
# }
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

def retrieve_tfidf(query: str, corpus: Optional[str], k: int = 5) -> List[Dict[str, Any]]:
    if not corpus or corpus not in CORPORA:
        return []
    c = CORPORA[corpus]
    tfidf: TfidfVectorizer = c["tfidf"]
    tfidf_mat = c["tfidf_mat"]
    docs = c["docs"]
    qv = tfidf.transform([query])                # (1, V)
    sims = (tfidf_mat @ qv.T).toarray().ravel()  # proxy similarity
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
    if not hits:
        return "No sources found."
    lines = []
    for i, h in enumerate(hits, start=1):
        label = f"[S{i}]"
        page = f"p{h['page']}" if h.get("page") else ""
        lines.append(f"{label} ({page} {h['id']}) {h['text']}")
    return "\n\n".join(lines)

# ========= Health / diag =========
@app.get("/")
def root():
    return {
        "ok": True,
        "message": "API is running",
        "endpoints": ["/chat (stream)", "/chat_json", "/diag", "/diag_rag", "/chat_rag_json", "/chat_sse"],
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

# ========= Non-stream JSON chat =========
@app.post("/chat_json")
async def chat_json(request: Request):
    try:
        body = await request.json()
        user_message = (body.get("message") or "").strip()
        override_religion = (body.get("religion") or "").strip().lower() or None
        if not user_message:
            return JSONResponse({"error": "message is required"}, status_code=400)

        sid = get_or_create_sid(request)
        detected = detect_religion(user_message)
        if detected:
            SESSIONS[sid]["religion"] = detected
        if override_religion in CORPORA:
            SESSIONS[sid]["religion"] = override_religion
        religion = SESSIONS[sid].get("religion")

        hits = retrieve_tfidf(user_message, religion, k=5) if religion else []
        context_block = build_sources_block(hits) if hits else "No sources found."
        grounded_prompt = (
            f"Use ONLY the sources below to answer. If they are insufficient, say so and ask for a more specific passage. "
            f"Cite inline as [S1], [S2], etc.\n\nSOURCES:\n{context_block}\n\nQUESTION:\n{user_message}"
            if hits else user_message
        )

        client = get_client()
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.3 if hits else 0.7,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": grounded_prompt},
            ],
        )
        answer = resp.choices[0].message.content

        r = JSONResponse({"response": answer, "sources": hits, "selected_corpus": religion})
        r.set_cookie("sid", sid, httponly=True, samesite="Lax")
        return r
    except Exception as e:
        traceback.print_exc()
        return JSONResponse({"error": str(e)}, status_code=500)

# ========= Streaming chat (text/plain) =========
@app.post("/chat")
async def chat_stream(request: Request):
    try:
        body = await request.json()
        user_message = (body.get("message") or "").strip()
        override_religion = (body.get("religion") or "").strip().lower() or None
        if not user_message:
            return JSONResponse({"error": "message is required"}, status_code=400)

        sid = get_or_create_sid(request)
        detected = detect_religion(user_message)
        if detected:
            SESSIONS[sid]["religion"] = detected
        if override_religion in CORPORA:
            SESSIONS[sid]["religion"] = override_religion
        religion = SESSIONS[sid].get("religion")

        hits = retrieve_tfidf(user_message, religion, k=5) if religion else []
        context_block = build_sources_block(hits) if hits else "No sources found."
        grounded_prompt = (
            f"Use ONLY the sources below to answer. If they are insufficient, say so and ask for a more specific passage. "
            f"Cite inline as [S1], [S2], etc.\n\nSOURCES:\n{context_block}\n\nQUESTION:\n{user_message}"
            if hits else user_message
        )

        client = get_client()

        def gen():
            stream = client.chat.completions.create(
                model="gpt-4o-mini",
                temperature=0.3 if hits else 0.7,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": grounded_prompt},
                ],
                stream=True,
            )
            for chunk in stream:
                delta = getattr(chunk.choices[0].delta, "content", None)
                if delta:
                    yield delta
            yield ""  # clean close

        resp = StreamingResponse(gen(), media_type="text/plain")
        resp.set_cookie("sid", sid, httponly=True, samesite="Lax")
        # helpful for proxies
        resp.headers["X-Accel-Buffering"] = "no"
        resp.headers["Cache-Control"] = "no-cache"
        return resp

    except Exception as e:
        traceback.print_exc()
        return JSONResponse({"error": str(e)}, status_code=500)

# ========= RAG chat (non-streaming JSON) =========
@app.post("/chat_rag_json")
async def chat_rag_json(request: Request):
    try:
        body = await request.json()
        user_message = (body.get("message") or "").strip()
        override_religion = (body.get("religion") or "").strip().lower() or None
        if not user_message:
            return JSONResponse({"error": "message is required"}, status_code=400)

        sid = get_or_create_sid(request)
        detected = detect_religion(user_message)
        if detected:
            SESSIONS[sid]["religion"] = detected
        if override_religion in CORPORA:
            SESSIONS[sid]["religion"] = override_religion
        religion = SESSIONS[sid].get("religion")

        hits = retrieve_tfidf(user_message, religion, k=5) if religion else []
        context_block = build_sources_block(hits) if hits else "No sources found."

        grounded_prompt = f"""Use ONLY the sources below to answer. If they are insufficient,
say so and ask for a more specific passage. Cite sources inline as [S1], [S2], etc.

SOURCES:
{context_block}

QUESTION:
{user_message}
"""

        client = get_client()
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.3 if hits else 0.7,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": grounded_prompt},
            ],
        )
        answer = resp.choices[0].message.content

        r = JSONResponse({"response": answer, "sources": hits, "selected_corpus": religion})
        r.set_cookie("sid", sid, httponly=True, samesite="Lax")
        return r
    except Exception as e:
        traceback.print_exc()
        return JSONResponse({"error": str(e)}, status_code=500)

# ========= SSE (Server-Sent Events) streaming =========
def _sse_event(data: str) -> bytes:
    return f"data: {data}\n\n".encode("utf-8")

@app.post("/chat_sse")
async def chat_sse(request: Request):
    try:
        body = await request.json()
        user_message = (body.get("message") or "").strip()
        override_religion = (body.get("religion") or "").strip().lower() or None
        if not user_message:
            return JSONResponse({"error": "message is required"}, status_code=400)

        sid = get_or_create_sid(request)
        detected = detect_religion(user_message)
        if detected:
            SESSIONS[sid]["religion"] = detected
        if override_religion in CORPORA:
            SESSIONS[sid]["religion"] = override_religion
        religion = SESSIONS[sid].get("religion")

        hits = retrieve_tfidf(user_message, religion, k=5) if religion else []
        context_block = build_sources_block(hits) if hits else "No sources found."
        grounded_note = (
            "Use ONLY the sources below to answer. If insufficient, say so and ask for a more specific passage. "
            "Cite inline as [S1], [S2], etc.\n\nSOURCES:\n" + context_block + "\n\nQUESTION:\n" + user_message
            if hits else user_message
        )

        client = get_client()

        def gen():
            # Initial comment: keeps some proxies happy
            yield b": connected\n\n"
            last_beat = time.time()

            stream = client.chat.completions.create(
                model="gpt-4o-mini",
                temperature=0.3 if hits else 0.7,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": grounded_note},
                ],
                stream=True,
            )
            for chunk in stream:
                delta = getattr(chunk.choices[0].delta, "content", None)
                if delta:
                    yield _sse_event(delta)
                    last_beat = time.time()

                # heartbeat every ~15s so the connection doesn't idle out
                now = time.time()
                if now - last_beat > 15:
                    yield b": keep-alive\n\n"
                    last_beat = now

            yield _sse_event("[DONE]")

        headers = {
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
        resp = StreamingResponse(gen(), media_type="text/event-stream", headers=headers)
        resp.set_cookie("sid", sid, httponly=True, samesite="Lax")
        return resp

    except Exception as e:
        traceback.print_exc()
        return JSONResponse({"error": str(e)}, status_code=500)
