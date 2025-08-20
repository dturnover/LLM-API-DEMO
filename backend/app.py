# app.py
import os
import re
import json
import uuid
import hashlib
from typing import Dict, List, Optional, Any, Iterable, Tuple
from pathlib import Path
from dataclasses import dataclass
from collections import defaultdict, Counter

from fastapi import FastAPI, Request, Response, Cookie
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

try:
    # Optional; only needed if you prefer EventSourceResponse for SSE
    from sse_starlette.sse import EventSourceResponse
    HAS_SSE_STARLETTE = True
except Exception:
    HAS_SSE_STARLETTE = False

# =========================
# Models
# =========================
class ChatIn(BaseModel):
    message: str

# =========================
# App & CORS
# =========================
app = FastAPI()

_FRONTEND_ORIGIN = os.getenv("FRONTEND_ORIGIN", "").strip()
_ALLOWED_ORIGINS = [o for o in [
    _ FRONTEND_ORIGIN if (_ORIGIN := _FRONTEND_ORIGIN) else None,  # keeps order
    "http://localhost:3000",
    "http://localhost:5173",
] if o]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_ALLOWED_ORIGINS or [],  # cannot be ["*"] when credentials=True
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# =========================
# In-memory state
# =========================
SESS: Dict[str, Dict[str, Any]] = {}     # sid -> { "faith": "christian" | "muslim" | "buddhist" }
MEM: Dict[str, Dict[str, str]] = {}      # sid -> normalized memory {norm(key): value}

# =========================
# Memory QoL helpers
# =========================
_WORDS = re.compile(r"[^a-z0-9 ]+")
def _norm(s: str) -> str:
    s = (s or "").lower().strip()
    s = " " + _WORDS.sub(" ", s) + " "
    for w in (" the ", " a ", " an ", " my ", " your "):
        s = s.replace(w, " ")
    return " ".join(s.split())

def mem_put(sid: str, key: str, value: str):
    d = MEM.setdefault(sid, {})
    d[_norm(key)] = value

def mem_get(sid: str, text: str) -> Optional[str]:
    t = _norm(text)
    for k, v in MEM.get(sid, {}).items():
        # direct containment or token overlap gives a hit
        if k in t:
            return v
        ktoks = set(k.split())
        ttoks = set(t.split())
        if ktoks and (len(ktoks & ttoks) / len(ktoks)) >= 0.6:
            return v
    return None

def parse_remember(msg: str) -> Optional[Tuple[str, str]]:
    """Try to extract (key, value) from 'remember ...' messages."""
    m = re.search(r"\bremember(?: that)?\s+(.+)", msg, flags=re.I)
    if not m:
        return None
    tail = m.group(1).strip()

    # Prefer patterns like "X is Y" / "X are Y"
    mi = re.match(r"(.+?)\s+(?:is|are|was|were)\s+(.+)", tail, flags=re.I)
    if mi:
        key = mi.group(1).strip()
        val = mi.group(2).strip().rstrip(".")
        return (key, val)

    # Fallback: if tail looks like "the penguin is painted blue" (already handled), else store entire phrase
    return (tail, "true")

# =========================
# Session / Cookies
# =========================
def new_sid() -> str:
    return uuid.uuid4().hex

def set_session_cookie(response: Response, request: Request, sid: str):
    origin = (request.headers.get("origin") or "").strip().lower()
    host = f"{request.url.scheme}://{request.url.netloc}".lower()
    cross_site = bool(origin and origin != host)

    if cross_site:
        response.set_cookie(
            key="sid",
            value=sid,
            path="/",
            httponly=True,
            secure=True,
            samesite="none",
        )
    else:
        response.set_cookie(
            key="sid",
            value=sid,
            path="/",
            httponly=True,
            secure=(request.url.scheme == "https"),
            samesite="lax",
        )

def get_or_create_sid(request: Request, response: Response) -> str:
    sid = request.cookies.get("sid")
    if not sid:
        sid = new_sid()
        set_session_cookie(response, request, sid)
    if sid not in SESS:
        SESS[sid] = {}
    return sid

# =========================
# Tiny RAG loader + search
# =========================
@dataclass
class Doc:
    id: str
    page: int
    text: str
    corpus: str

CORPUS_DOCS: Dict[str, List[Doc]] = defaultdict(list)
CORPUS_COUNTS: Dict[str, int] = defaultdict(int)

def _load_jsonl(p: Path) -> Iterable[Dict[str, Any]]:
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue

def _load_json(p: Path) -> Iterable[Dict[str, Any]]:
    with p.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        for x in data:
            yield x
    elif isinstance(data, dict):
        # maybe wrapped
        if "items" in data and isinstance(data["items"], list):
            for x in data["items"]:
                yield x

def _ingest_record(rec: Dict[str, Any]):
    text = (rec.get("text") or "").strip()
    if not text:
        return
    corpus = (rec.get("corpus") or "").strip().lower() or "misc"
    rid = rec.get("id") or f"{corpus}_{rec.get('page', 0)}"
    page = int(rec.get("page") or 0)
    CORPUS_DOCS[corpus].append(Doc(id=rid, page=page, text=text, corpus=corpus))

def load_corpora():
    base = Path(__file__).parent / "rag_data"
    if not base.exists():
        return
    for p in base.glob("*.jsonl"):
        try:
            for rec in _load_jsonl(p):
                _ingest_record(rec)
        except Exception:
            pass
    for p in base.glob("*.json"):
        try:
            for rec in _load_json(p):
                _ingest_record(rec)
        except Exception:
            pass
    for k, arr in CORPUS_DOCS.items():
        CORPUS_COUNTS[k] = len(arr)

load_corpora()

def detect_corpus(msg: str, faith: Optional[str]) -> Optional[str]:
    m = msg.lower()
    if "qur" in m or "koran" in m or "quran" in m:
        return "quran"
    if "bible" in m or "psalm" in m or "jesus" in m or "[s#" in m:
        return "bible"
    if "sutta" in m or "pitaka" in m or "buddha" in m:
        return "sutta_pitaka"
    if faith == "christian":
        return "bible"
    if faith == "muslim":
        return "quran"
    if faith == "buddhist":
        return "sutta_pitaka"
    return None

def simple_score(text: str, toks: List[str]) -> int:
    t = text.lower()
    return sum(t.count(tok) for tok in toks if tok)

def rag_search(query: str, corpus: str, k: int = 3) -> List[Doc]:
    docs = CORPUS_DOCS.get(corpus, [])
    if not docs:
        return []
    # crude keyword scoring
    q = query.lower()
    # strip [S#] tags and filler words
    q = re.sub(r"\[s#\]", " ", q, flags=re.I)
    q = re.sub(r"[^a-z0-9 ]+", " ", q)
    toks = [w for w in q.split() if w not in {"give", "share", "short", "about", "verse", "a", "the", "of", "and"}]
    if not toks and query.strip():
        toks = query.lower().split()
    scored = [(simple_score(d.text, toks), d) for d in docs]
    scored.sort(key=lambda x: x[0], reverse=True)
    top = [d for s, d in scored[:k] if s > 0] or [docs[0]]
    return top

def build_sources(docs: List[Doc]) -> List[Dict[str, Any]]:
    out = []
    for d in docs[:5]:
        # keep a small text snippet
        snippet = d.text
        if len(snippet) > 900:
            snippet = snippet[:900] + "..."
        out.append({
            "score": 0.0,
            "id": d.id,
            "page": d.page,
            "text": snippet,
            "corpus": d.corpus
        })
    return out

# =========================
# Health & Diag
# =========================
@app.get("/health")
def health():
    return {"ok": True}

@app.get("/diag_rag")
def diag_rag():
    return {"ok": True, "corpora": CORPUS_COUNTS}

# =========================
# Chat Logic
# =========================
FAITH_MAP = {
    "christian": "bible",
    "muslim": "quran",
    "buddhist": "sutta_pitaka",
}

def maybe_set_faith(message: str, sid: str) -> Optional[str]:
    m = message.lower().strip()
    # very light detection: "i am X", "i'm X"
    mi = re.search(r"\b(i am|i'm)\s+(christian|muslim|buddhist)\b", m)
    if mi:
        faith = mi.group(2)
        SESS.setdefault(sid, {})["faith"] = faith
        return faith
    return None

def format_memory_answer(key: str, value: str) -> str:
    # attempt a natural sentence if value looks like "painted blue" or "red"
    if re.match(r"^(painted\s+)?[a-z ]+$", value):
        v = value.strip()
        if not v.startswith(("is ", "are ")):
            return f"{key.strip()} is {v}."
        return f"{key.strip()} {v}."
    return f"{key.strip()}: {value}"

def answer_json(message: str, selected_corpus: Optional[str], sources: List[Dict[str, Any]], remembered: Optional[Dict[str, str]], body_text: str) -> JSONResponse:
    payload = {
        "response": body_text,
        "sources": sources or [],
        "selected_corpus": selected_corpus,
        "remembered": remembered,
    }
    return JSONResponse(payload)

def make_generic_reply(message: str) -> str:
    # fallback, simple helpful tone (you can swap to an OpenAI call if desired)
    return "How can I help further?"

# =========================
# /chat (JSON) endpoint
# =========================
@app.post("/chat")
async def chat(payload: ChatIn, request: Request):
    response = Response(media_type="application/json")
    sid = get_or_create_sid(request, response)
    msg = payload.message.strip()

    # 1) Faith detection
    new_faith = maybe_set_faith(msg, sid)

    # 2) Memory store
    remembered: Optional[Dict[str, str]] = None
    if re.search(r"\bremember\b", msg, flags=re.I):
        kv = parse_remember(msg)
        if kv:
            key, val = kv
            mem_put(sid, key, val)
            remembered = {"key": key, "value": val}

    # 3) Memory recall (fast path)
    recalled = mem_get(sid, msg)
    if recalled:
        # try to guess the subject to make a nice sentence
        subj = None
        # find something that matches a stored key
        for k in MEM.get(sid, {}):
            if k in _norm(msg):
                subj = k.strip()
                break
        text = format_memory_answer(subj or "That", recalled)
        jr = answer_json(msg, None, [], None, text)
        # transfer our cookies header to JSONResponse
        for k, v in response.headers.items():
            jr.headers[k] = v
        return jr

    # 4) Scripture / RAG
    faith = SESS.get(sid, {}).get("faith")
    sel = detect_corpus(msg, faith)
    if sel:
        docs = rag_search(msg, sel, k=3)
        srcs = build_sources(docs)
        # Build a short quotation line with [S1]
        quote = docs[0].text.strip().split("\n")[0]
        if len(quote) > 240:
            quote = quote[:240].rsplit(" ", 1)[0] + "…"
        reply = f"{quote} [S1]"
        jr = answer_json(msg, sel, srcs, remembered, reply)
        for k, v in response.headers.items():
            jr.headers[k] = v
        return jr

    # 5) Generic
    jr = answer_json(msg, None, [], remembered, make_generic_reply(msg))
    for k, v in response.headers.items():
        jr.headers[k] = v
    return jr

# =========================
# /chat_sse (streaming)
# =========================
def stream_tokens(text: str) -> Iterable[str]:
    # simple word streaming to emulate token flow
    for w in re.split(r"(\s+)", text):
        if not w:
            continue
        yield w

def sse_headers() -> Dict[str, str]:
    return {
        "Cache-Control": "no-cache, no-transform",
        "X-Accel-Buffering": "no",
        "Connection": "keep-alive",
    }

@app.post("/chat_sse")
async def chat_sse(payload: ChatIn, request: Request):
    # We still want to set/refresh cookies if a new sid is minted.
    sid_resp = Response()
    sid = get_or_create_sid(request, sid_resp)
    msg = payload.message.strip()

    # prepare response fields
    faith = maybe_set_faith(msg, sid) or SESS.get(sid, {}).get("faith")
    remembered: Optional[Dict[str, str]] = None
    sources: List[Dict[str, Any]] = []
    selected_corpus: Optional[str] = None

    # memory store
    if re.search(r"\bremember\b", msg, flags=re.I):
        kv = parse_remember(msg)
        if kv:
            key, val = kv
            mem_put(sid, key, val)
            remembered = {"key": key, "value": val}

    # recall
    recalled = mem_get(sid, msg)
    if recalled:
        subj = None
        for k in MEM.get(sid, {}):
            if k in _norm(msg):
                subj = k.strip()
                break
        full_text = format_memory_answer(subj or "That", recalled)
    else:
        # RAG?
        selected_corpus = detect_corpus(msg, faith)
        if selected_corpus:
            docs = rag_search(msg, selected_corpus, k=3)
            sources = build_sources(docs)
            quote = docs[0].text.strip().split("\n")[0]
            if len(quote) > 240:
                quote
