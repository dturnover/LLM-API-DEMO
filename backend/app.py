import os, json, re, uuid, traceback
from pathlib import Path
from typing import Dict, Any, List, Optional

import numpy as np
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse, Response, PlainTextResponse
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

MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
MAX_HISTORY_TURNS = 12  # user+assistant pairs kept in session
MAX_STICKY = 30         # max sticky notes to remember

# ========= CORS =========
origins = [
    "https://dturnover.github.io",  # GitHub Pages demo
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
You are “Winston Churchill GPT,” a conversational assistant speaking in the voice and manner of Sir Winston Churchill.

Voice & Style:
- Formal, eloquent, oratorical; occasionally wry.
- Uses period-appropriate phrasing; avoid anachronisms and modern slang.
- Concise for direct questions; expansive when invited to opine.

Behavior:
- If unclear, ask a brief clarifying question.
- If asked about post-1965 events, respond hypothetically from Churchill’s perspective, and note the limitation.
- Be courteous and refuse harmful requests.

Grounding:
- When a “Sources” block is provided, treat it as authoritative context for this reply and, when quoting, cite inline as [S1], [S2], etc. If insufficient, say so briefly.
- When “Sticky Notes” are provided, treat them as facts remembered about the current user; use them naturally if relevant (do not cite stickies).

Stay in character throughout.
""".strip()

# ========= Session store (in-memory per process) =========
# sid -> {
#   "religion": Optional[str],
#   "sticky": Dict[str, str],
#   "history": List[Dict[str, str]]  # [{"role":"user"/"assistant","content": "..."}]
# }
SESSIONS: Dict[str, Dict[str, Any]] = {}

def get_or_create_sid(request: Request) -> str:
    sid = request.cookies.get("sid")
    if not sid or sid not in SESSIONS:
        sid = uuid.uuid4().hex
        SESSIONS[sid] = {"religion": None, "sticky": {}, "history": []}
    return sid

def add_history(sid: str, role: str, content: str):
    h = SESSIONS[sid]["history"]
    h.append({"role": role, "content": content})
    # keep last MAX_HISTORY_TURNS * 2 messages (user+assistant pairs)
    max_msgs = MAX_HISTORY_TURNS * 2
    if len(h) > max_msgs:
        del h[: len(h) - max_msgs]

# ========= Religion detection =========
RELIGION_MAP = {
    "bible": "bible", "christian": "bible", "christianity": "bible", "jesus": "bible", "gospel": "bible",
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

# ========= Sticky Notes (simple memory) =========
def normalize_key(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"[^\w\s-]", "", s)
    s = re.sub(r"\s+", " ", s)
    return s[:120]

REMEMBER_PATTERNS = [
    re.compile(r"^\s*remember that\s+(.+?)\s+(?:is|=|:)\s+(.+?)\s*$", re.I),
    re.compile(r"^\s*remember\s+(.+?)\s+(?:is|=|:)\s+(.+?)\s*$", re.I),
    re.compile(r"^\s*please remember\s+(.+?)\s+(?:is|=|:)\s+(.+?)\s*$", re.I),
]

def extract_sticky(message: str) -> Optional[tuple[str, str]]:
    for pat in REMEMBER_PATTERNS:
        m = pat.match(message)
        if m:
            key = normalize_key(m.group(1))
            val = m.group(2).strip()
            # simple guardrails (avoid storing long/injected blobs)
            if 1 <= len(key) <= 120 and 1 <= len(val) <= 200:
                return key, val
    return None

def render_sticky_block(sticky: Dict[str, str]) -> str:
    if not sticky:
        return "Sticky Notes: (none)"
    lines = [f"- {k}: {v}" for k, v in sticky.items()]
    return "Sticky Notes:\n" + "\n".join(lines)

# ========= RAG load (multi-corpus) =========
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
    if not hits:
        return "Sources: (none)"
    lines = []
    for i, h in enumerate(hits, start=1):
        page = f"p{h['page']}" if h.get("page") else ""
        lines.append(f"[S{i}] ({page} {h['id']}) {h['text']}")
    return "Sources:\n" + "\n\n".join(lines)

# ========= Helpers =========
def build_messages(sid: str, user_text: str, hits: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    sticky_block = render_sticky_block(SESSIONS[sid]["sticky"])
    sources_block = build_sources_block(hits)
    # Compose:
    msgs: List[Dict[str, str]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "system", "content": sticky_block},
        {"role": "system", "content": sources_block},
    ]
    # history
    msgs.extend(SESSIONS[sid]["history"])
    # current user
    msgs.append({"role": "user", "content": user_text})
    return msgs

def select_hits_for_message(sid: str, message: str) -> List[Dict[str, Any]]:
    corpus = SESSIONS[sid].get("religion")
    if not corpus:
        # opportunistically set religion if message hints it
        maybe = detect_religion(message)
        if maybe:
            SESSIONS[sid]["religion"] = maybe
            corpus = maybe
    # Heuristic: retrieve if a corpus is set AND the message looks like it could use scripture,
    # or if the user explicitly asks for verse/scripture.
    if not corpus:
        return []
    triggers = ("verse", "scripture", "quote", "bible", "quran", "sutta", "sutra", "teachings", "patience", "courage", "fear", "hope", "strength")
    needs = any(w in message.lower() for w in triggers)
    # Also retrieve if prior assistant used a citation recently
    recent_cited = any("[S" in (m["content"] or "") for m in SESSIONS[sid]["history"][-4:] if m["role"] == "assistant")
    if needs or recent_cited:
        return retrieve_tfidf(message, corpus, k=5)
    return []

def sse_encode(event: Optional[str], data: str) -> bytes:
    # Server-Sent Events framing
    out = []
    if event:
        out.append(f"event: {event}")
    # split long lines
    for line in data.splitlines() or [""]:
        out.append(f"data: {line}")
    out.append("")  # end of message
    return ("\n".join(out) + "\n").encode("utf-8")

# ========= Endpoints =========
@app.get("/")
def root():
    return {
        "ok": True,
        "message": "API is running",
        "endpoints": ["/chat_json", "/chat_sse", "/diag", "/diag_rag"]
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

# ---- JSON (non-stream) ----
@app.post("/chat_json")
async def chat_json(request: Request):
    try:
        body = await request.json()
        user_text: str = (body.get("message") or "").strip()
        if not user_text:
            return JSONResponse({"error": "message is required"}, status_code=400)

        sid = get_or_create_sid(request)

        # Sticky “remember …”?
        remembered = None
        kv = extract_sticky(user_text)
        if kv:
            k, v = kv
            sticky = SESSIONS[sid]["sticky"]
            if len(sticky) >= MAX_STICKY and k not in sticky:
                # drop oldest by insertion order
                oldest_key = next(iter(sticky.keys()))
                sticky.pop(oldest_key, None)
            sticky[k] = v
            remembered = {"key": k, "value": v}

        # Religion override from body?
        chosen = body.get("religion")
        if isinstance(chosen, str) and chosen.lower() in CORPORA:
            SESSIONS[sid]["religion"] = chosen.lower()

        # Auto-detect religion if user hints it
        maybe = detect_religion(user_text)
        if maybe:
            SESSIONS[sid]["religion"] = maybe

        hits = select_hits_for_message(sid, user_text)
        msgs = build_messages(sid, user_text, hits)

        client = get_client()
        resp = client.chat.completions.create(
            model=MODEL,
            messages=msgs,
            temperature=0.7,
        )
        text = resp.choices[0].message.content or ""

        # update history
        add_history(sid, "user", user_text)
        add_history(sid, "assistant", text)

        payload = {"response": text, "sources": hits, "selected_corpus": SESSIONS[sid]["religion"], "remembered": remembered}
        r = JSONResponse(payload)
        # ensure cookie is set
        if not request.cookies.get("sid"):
            r.set_cookie("sid", sid, httponly=True, samesite="lax", path="/")
        return r
    except Exception as e:
        traceback.print_exc()
        return JSONResponse({"error": str(e)}, status_code=500)

# ---- SSE streaming ----
@app.post("/chat_sse")
async def chat_sse(request: Request):
    try:
        body = await request.json()
        user_text: str = (body.get("message") or "").strip()
        if not user_text:
            return PlainTextResponse("Bad Request: message is required", status_code=400)

        sid = get_or_create_sid(request)

        # Sticky remember?
        remembered = None
        kv = extract_sticky(user_text)
        if kv:
            k, v = kv
            sticky = SESSIONS[sid]["sticky"]
            if len(sticky) >= MAX_STICKY and k not in sticky:
                oldest_key = next(iter(sticky.keys()))
                sticky.pop(oldest_key, None)
            sticky[k] = v
            remembered = {"key": k, "value": v}

        # Optional religion override in payload
        chosen = body.get("religion")
        if isinstance(chosen, str) and chosen.lower() in CORPORA:
            SESSIONS[sid]["religion"] = chosen.lower()

        # Auto-detect
        maybe = detect_religion(user_text)
        if maybe:
            SESSIONS[sid]["religion"] = maybe

        hits = select_hits_for_message(sid, user_text)
        msgs = build_messages(sid, user_text, hits)

        client = get_client()
        stream = client.chat.completions.create(
            model=MODEL,
            messages=msgs,
            temperature=0.7,
            stream=True,
        )

        def gen():
            # initial ping
            yield sse_encode(None, ": connected").decode("utf-8").encode("utf-8")
            try:
                full = []
                for chunk in stream:
                    for choice in chunk.choices:
                        delta = getattr(choice, "delta", None)
                        if delta and delta.content:
                            full.append(delta.content)
                            yield sse_encode(None, delta.content)
                # done
                yield sse_encode(None, "[DONE]")
                # sources event (after content so UIs can render)
                yield sse_encode("sources", json.dumps({
                    "sources": hits,
                    "selected_corpus": SESSIONS[sid]["religion"],
                    "remembered": remembered
                }))
            except Exception as ex:
                err = f"{ex}"
                yield sse_encode(None, json.dumps({"error": err}))
            finally:
                # update history
                try:
                    final_text = "".join(full) if full else ""
                    add_history(sid, "user", user_text)
                    if final_text:
                        add_history(sid, "assistant", final_text)
                except Exception:
                    pass

        resp = StreamingResponse(gen(), media_type="text/event-stream; charset=utf-8")
        if not request.cookies.get("sid"):
            resp.set_cookie("sid", sid, httponly=True, samesite="lax", path="/")
        return resp
    except Exception as e:
        traceback.print_exc()
        return JSONResponse({"error": str(e)}, status_code=500)
