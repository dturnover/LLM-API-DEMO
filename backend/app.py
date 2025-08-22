# app.py
import os, re, json, uuid
from typing import Dict, List, Optional, Any, Iterable, Tuple
from pathlib import Path
from dataclasses import dataclass
from collections import defaultdict

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

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
    _FRONTEND_ORIGIN or None,
    "http://localhost:3000",
    "http://localhost:5173",
] if o]

_ALLOW_ANY = os.getenv("CORS_ALLOW_ANY", "0") == "1"

app.add_middleware(
    CORSMiddleware,
    allow_origins=[] if _ALLOW_ANY else _ALLOWED_ORIGINS,
    allow_origin_regex=".*" if _ALLOW_ANY else None,
    allow_credentials=True,                         # allow cookies
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# =========================
# In-memory state
# =========================
SESS: Dict[str, Dict[str, Any]] = {}     # sid -> { "faith": ... }
MEM: Dict[str, Dict[str, str]] = {}      # sid -> normalized memory

# =========================
# Memory helpers
# =========================
_WORDS = re.compile(r"[^a-z0-9 ]+")
def _norm(s: str) -> str:
    s = (s or "").lower().strip()
    s = " " + _WORDS.sub(" ", s) + " "
    for w in (" the ", " a ", " an ", " my ", " your "):
        s = s.replace(w, " ")
    return " ".join(s.split())

def mem_put(sid: str, key: str, value: str):
    MEM.setdefault(sid, {})[_norm(key)] = value

def mem_get(sid: str, text: str) -> Optional[str]:
    t = _norm(text)
    for k, v in MEM.get(sid, {}).items():
        if k in t:
            return v
        ktoks = set(k.split()); ttoks = set(t.split())
        if ktoks and (len(ktoks & ttoks) / len(ktoks)) >= 0.6:
            return v
    return None

def parse_remember(msg: str) -> Optional[Tuple[str, str]]:
    m = re.search(r"\bremember(?: that)?\s+(.+)", msg, flags=re.I)
    if not m: return None
    tail = m.group(1).strip()
    mi = re.match(r"(.+?)\s+(?:is|are|was|were)\s+(.+)", tail, flags=re.I)
    if mi: return (mi.group(1).strip(), mi.group(2).strip().rstrip("."))
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
    response.set_cookie(
        key="sid", value=sid, path="/", httponly=True,
        secure=True if cross_site else (request.url.scheme == "https"),
        samesite="none" if cross_site else "lax",
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
# Tiny RAG: load + search
# =========================
@dataclass
class Doc:
    id: str
    page: int
    text: str
    corpus: str

CORPUS_DOCS: Dict[str, List[Doc]] = defaultdict(list)
CORPUS_COUNTS: Dict[str, int] = defaultdict(int)

def _load_jsonl(p: Path):
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try: yield json.loads(line)
            except Exception: continue

def _load_json(p: Path):
    with p.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        for x in data: yield x
    elif isinstance(data, dict) and isinstance(data.get("items"), list):
        for x in data["items"]: yield x

def _ingest_record(rec: Dict[str, Any]):
    text = (rec.get("text") or "").strip()
    if not text: return
    corpus = (rec.get("corpus") or "").strip().lower() or "misc"
    rid = rec.get("id") or f"{corpus}_{rec.get('page', 0)}"
    page = int(rec.get("page") or 0)
    CORPUS_DOCS[corpus].append(Doc(id=rid, page=page, text=text, corpus=corpus))

def load_corpora():
    base = Path(__file__).parent / "rag_data"
    if not base.exists(): return
    # root files
    for p in list(base.glob("*.jsonl")) + list(base.glob("*.json")):
        try:
            loader = _load_jsonl if p.suffix.lower()==".jsonl" else _load_json
            for rec in loader(p): _ingest_record(rec)
        except Exception: pass
    # per-corpus subfolders (prep_rag.py writes here)
    for sub in base.iterdir():
        if not sub.is_dir(): continue
        dj = sub / "docs.json"
        if dj.exists():
            try:
                for rec in _load_json(dj):
                    rec.setdefault("corpus", sub.name)
                    _ingest_record(rec)
            except Exception: pass
    for k, arr in CORPUS_DOCS.items():
        CORPUS_COUNTS[k] = len(arr)

load_corpora()

def detect_corpus(msg: str, faith: Optional[str]) -> Optional[str]:
    m = msg.lower()
    if "qur" in m or "koran" in m or "quran" in m: return "quran"
    if "bible" in m or "psalm" in m or "jesus" in m or "[s#" in m: return "bible"
    if "sutta" in m or "pitaka" in m or "buddha" in m: return "sutta_pitaka"
    if faith == "christian": return "bible"
    if faith == "muslim": return "quran"
    if faith == "buddhist": return "sutta_pitaka"
    return None

def simple_score(text: str, toks: List[str]) -> int:
    t = text.lower()
    return sum(t.count(tok) for tok in toks if tok)

def rag_search(query: str, corpus: str, k: int = 3) -> List[Doc]:
    docs = CORPUS_DOCS.get(corpus, [])
    if not docs: return []
    q = re.sub(r"\[s#\]", " ", query.lower(), flags=re.I)
    q = re.sub(r"[^a-z0-9 ]+", " ", q)
    toks = [w for w in q.split() if w not in {"give","share","short","about","verse","a","the","of","and"}] or q.split()
    scored = [(simple_score(d.text, toks), d) for d in docs]
    scored.sort(key=lambda x: x[0], reverse=True)
    return [d for s, d in scored[:k] if s > 0] or docs[:1]

def build_sources(docs: List[Doc]) -> List[Dict[str, Any]]:
    out = []
    for d in docs[:5]:
        snippet = d.text if len(d.text) <= 900 else d.text[:900] + "..."
        out.append({"score": 0.0, "id": d.id, "page": d.page, "text": snippet, "corpus": d.corpus})
    return out

# =========================
# Health & Diag
# =========================
@app.get("/health")
def health():
    return {"ok": True, "sessions": len(SESS), "corpora": CORPUS_COUNTS}

@app.get("/diag_rag")
def diag_rag(reload: Optional[int] = 0):
    if reload:
        CORPUS_DOCS.clear(); CORPUS_COUNTS.clear()
        load_corpora()
    return {"ok": True, "corpora": CORPUS_COUNTS}

# =========================
# Chat Logic
# =========================
def maybe_set_faith(message: str, sid: str) -> Optional[str]:
    m = message.lower().strip()
    mi = re.search(r"\b(i am|i'm)\s+(christian|muslim|buddhist)\b", m)
    if mi:
        faith = mi.group(2)
        SESS.setdefault(sid, {})["faith"] = faith
        return faith
    return None

def format_memory_answer(key: str, value: str) -> str:
    if re.match(r"^(painted\s+)?[a-z ]+$", value):
        v = value.strip()
        if not v.startswith(("is ", "are ")):
            return f"{key.strip()} is {v}."
        return f"{key.strip()} {v}."
    return f"{key.strip()}: {value}"

def answer_json(message: str, selected_corpus: Optional[str], sources: List[Dict[str, Any]], remembered: Optional[Dict[str, str]], body_text: str) -> JSONResponse:
    return JSONResponse({
        "response": body_text,
        "sources": sources or [],
        "selected_corpus": selected_corpus,
        "remembered": remembered,
    })

def make_generic_reply(_: str) -> str:
    return "How can I help further?"

# =========================
# /chat (JSON)
# =========================
@app.post("/chat")
async def chat(payload: ChatIn, request: Request):
    response = Response(media_type="application/json")
    sid = get_or_create_sid(request, response)
    msg = payload.message.strip()

    maybe_set_faith(msg, sid)

    remembered: Optional[Dict[str, str]] = None
    if re.search(r"\bremember\b", msg, flags=re.I):
        kv = parse_remember(msg)
        if kv:
            key, val = kv
            mem_put(sid, key, val)
            remembered = {"key": key, "value": val}

    recalled = mem_get(sid, msg)
    if recalled:
        subj = next((k.strip() for k in MEM.get(sid, {}) if k in _norm(msg)), None)
        text = format_memory_answer(subj or "That", recalled)
        jr = answer_json(msg, None, [], None, text)
        for k, v in response.headers.items(): jr.headers[k] = v
        return jr

    faith = SESS.get(sid, {}).get("faith")
    sel = detect_corpus(msg, faith)
    if sel:
        docs = rag_search(msg, sel, k=3)
        srcs = build_sources(docs)
        quote = (docs[0].text.strip().split("\n")[0] if docs else "No passage found.")
        if len(quote) > 240: quote = quote[:240].rsplit(" ", 1)[0] + "…"
        reply = f"{quote} [S1]"
        jr = answer_json(msg, sel, srcs, remembered, reply)
        for k, v in response.headers.items(): jr.headers[k] = v
        return jr

    jr = answer_json(msg, None, [], remembered, make_generic_reply(msg))
    for k, v in response.headers.items(): jr.headers[k] = v
    return jr

# =========================
# /chat_sse (StreamingResponse only)
# =========================
def stream_tokens(text: str) -> Iterable[str]:
    for w in re.split(r"(\s+)", text):
        if w: yield w

@app.post("/chat_sse")
async def chat_sse(payload: ChatIn, request: Request):
    sid_resp = Response()
    sid = get_or_create_sid(request, sid_resp)
    msg = payload.message.strip()

    faith = maybe_set_faith(msg, sid) or SESS.get(sid, {}).get("faith")
    remembered: Optional[Dict[str, str]] = None
    sources: List[Dict[str, Any]] = []
    selected_corpus: Optional[str] = None

    if re.search(r"\bremember\b", msg, flags=re.I):
        kv = parse_remember(msg)
        if kv:
            key, val = kv
            mem_put(sid, key, val)
            remembered = {"key": key, "value": val}

    recalled = mem_get(sid, msg)
    if recalled:
        subj = next((k.strip() for k in MEM.get(sid, {}) if k in _norm(msg)), None)
        full_text = format_memory_answer(subj or "That", recalled)
    else:
        selected_corpus = detect_corpus(msg, faith)
        if selected_corpus:
            docs = rag_search(msg, selected_corpus, k=3)
            sources = build_sources(docs)
            quote = (docs[0].text.strip().split("\n")[0] if docs else "No passage found.")
            if len(quote) > 240: quote = quote[:240].rsplit(" ", 1)[0] + "…"
            full_text = f"{quote} [S1]"
        else:
            full_text = make_generic_reply(msg)

    async def gen():
        # 1) open stream immediately so Content-Length isn't 0
        yield ": connected\n\n"
        # 2) tokens
        for tok in stream_tokens(full_text):
            yield f"data: {tok}\n\n"
        # 3) metadata event
        meta = {"sources": sources, "selected_corpus": selected_corpus, "remembered": remembered}
        yield f"event: sources\ndata: {json.dumps(meta)}\n\n"
        # 4) done event
        yield "event: done\ndata: {}\n\n"

    headers = {
        "Cache-Control": "no-cache, no-transform",
        "X-Accel-Buffering": "no",
        "Connection": "keep-alive",
    }
    headers.update(sid_resp.headers)
    return StreamingResponse(gen(), media_type="text/event-stream", headers=headers)

# =========================
# Dev: run
# =========================
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)
