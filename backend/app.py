# app.py
import os, re, json, uuid
from typing import Dict, List, Optional, Any, Tuple, Generator

from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

# ---- OpenAI (>=1.0) ----
try:
    from openai import OpenAI
    _HAS_OPENAI = True
except Exception:
    _HAS_OPENAI = False
    OpenAI = None  # type: ignore

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL   = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# ========= Session state (in-memory) =========
class SessionState:
    def __init__(self) -> None:
        self.faith: Optional[str] = None        # "bible" | "quran" | "sutta_pitaka"
        self.facts: Dict[str, str] = {}         # remembered facts
        self.history: List[Dict[str, str]] = [] # short rolling window

SESSIONS: Dict[str, SessionState] = {}

def get_or_create_sid(request: Request, response: Response) -> Tuple[str, SessionState]:
    sid = request.cookies.get("sid")
    if not sid:
        sid = uuid.uuid4().hex
        # cross-site cookie (Pages -> Render)
        response.set_cookie(
            "sid", sid, httponly=True, samesite="none", secure=True, path="/"
        )
    if sid not in SESSIONS:
        SESSIONS[sid] = SessionState()
    return sid, SESSIONS[sid]

# ========= RAG (very small + fast) =========
from pathlib import Path
RAG_ROOT = Path(__file__).parent / "rag_data"
CORPUS_FILES = {"bible":"bible.jsonl", "quran":"quran.jsonl", "sutta_pitaka":"sutta_pitaka.jsonl"}
_LOADED: Dict[str, List[Dict[str, Any]]] = {}

def _load_corpus(corpus: str) -> List[Dict[str, Any]]:
    if corpus not in CORPUS_FILES: return []
    if corpus in _LOADED: return _LOADED[corpus]
    p = RAG_ROOT / CORPUS_FILES[corpus]
    docs: List[Dict[str, Any]] = []
    if p.exists():
        with p.open("r", encoding="utf-8", errors="ignore") as f:
            for i, line in enumerate(f):
                try:
                    obj = json.loads(line)
                    obj.setdefault("text", obj.get("text") or obj.get("content") or "")
                    obj.setdefault("page", obj.get("page") or i)
                    obj["id"] = obj.get("id") or f"p{obj['page']}_c{i}"
                    obj["corpus"] = corpus
                    docs.append(obj)
                except Exception:
                    continue
    _LOADED[corpus] = docs
    return docs

def diag_rag_counts() -> Dict[str,int]:
    return {c: len(_load_corpus(c)) for c in CORPUS_FILES}

def search_scripture(query: str, corpus: str, k: int = 3) -> Tuple[List[Dict[str,Any]], Optional[str]]:
    docs = _load_corpus(corpus)
    if not docs: return [], None
    terms = [w for w in re.sub(r"[^a-z0-9\s]", " ", query.lower()).split() if len(w)>2]
    scored = []
    for d in docs:
        t = d.get("text","").lower()
        score = sum(t.count(w) for w in terms)
        if score>0: scored.append((score,d))
    if not scored: return [], None
    scored.sort(key=lambda x:x[0], reverse=True)
    tops = [d for _,d in scored[:k]]
    best = tops[0].get("text","")
    # return first sentence-ish as a "verse-ish" pull, capped
    verse = re.split(r"(?<=[.!?])\s+", best.strip())[0]
    if len(verse)>220:
        verse = verse[:200].rsplit(" ",1)[0]+"..."
    return tops, f"{verse} [S1]"

# ========= Heuristics =========
FAITH_MAP = {
    "christian": "bible", "christianity":"bible", "bible":"bible",
    "muslim":"quran", "islam":"quran", "qur":"quran", "quran":"quran",
    "buddhist":"sutta_pitaka", "buddhism":"sutta_pitaka", "sutta":"sutta_pitaka",
}
def try_set_faith(message: str, s: SessionState) -> None:
    m = message.lower()
    for k,v in FAITH_MAP.items():
        if re.search(rf"\b{re.escape(k)}\b", m): s.faith=v; return

REMEMBER_RE = re.compile(r"\bremember\b", re.I)
IS_RE       = re.compile(r"^\s*(?P<k>.+?)\s+(?:is|are|=)\s+(?P<v>.+?)\s*$", re.I)
def normalize_key(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"^(the|a|an|this|that)\s+", "", s)
    return re.sub(r"\s+"," ", s)

def parse_and_remember(message: str, s: SessionState) -> Optional[Dict[str,str]]:
    if not REMEMBER_RE.search(message): return None
    body = re.sub(r"^\s*remember\s*(that|:)?\s*", "", message, flags=re.I).strip()
    m = IS_RE.match(body)
    if m:
        k = normalize_key(m.group("k")); v = m.group("v").strip()
        if k: s.facts[k]=v; return {"key":k,"value":v}
    # fallback: first 4 words = key
    words = body.split()
    k = normalize_key(" ".join(words[:4])) or "note"
    v = " ".join(words[4:]) or body
    s.facts[k]=v
    return {"key":k,"value":v}

Q_COLOR = re.compile(r"^\s*what\s+(color|colour)\s+is\s+(?:the\s+)?(?P<x>[^?!.]+)\??\s*$", re.I)
Q_WHAT  = re.compile(r"^\s*what\s+is\s+(?:the\s+)?(?P<x>[^?!.]+)\??\s*$", re.I)
def lookup_memory_answer(message: str, s: SessionState) -> Optional[str]:
    m = Q_COLOR.match(message) or Q_WHAT.match(message)
    if not m: return None
    x = normalize_key(m.group("x"))
    return s.facts.get(x) or s.facts.get(x.replace("the ","",1), None)

def wants_scripture(message: str) -> bool:
    m = message.lower()
    return any(w in m for w in ["verse","scripture","[s#]"])

def is_emotional(message: str) -> bool:
    m = message.lower()
    hot = ["scared","afraid","anxious","nervous","panic","breakup","hurt","down",
           "lost","depressed","angry","frustrated","worried","fight","injury","pain"]
    return any(w in m for w in hot)

def detect_corpus(message: str, s: SessionState) -> str:
    m = message.lower()
    if "bible" in m or "christ" in m: return "bible"
    if "qur" in m or "islam" in m or "muslim" in m: return "quran"
    if "sutta" in m or "buddh" in m: return "sutta_pitaka"
    return s.faith or "bible"

def build_system_prompt(s: SessionState) -> str:
    lines = [
        "You are Fight Chaplain: calm, concise, and encouraging.",
        "Offer practical corner-coach guidance with spiritual grounding.",
        "When a verse is appropriate, quote plainly and append a bracket citation like [S1].",
        "Keep answers tight; avoid flowery language.",
    ]
    if s.faith: lines.append(f"User faith: {s.faith}. Prefer that corpus.")
    if s.facts:
        lines.append("STICKY_NOTES:")
        for k,v in list(s.facts.items())[:30]:
            lines.append(f"- {k} = {v}")
    return "\n".join(lines)

def _build_messages(s: SessionState, user: str) -> List[Dict[str,str]]:
    msgs = [{"role":"system","content":build_system_prompt(s)}]
    msgs += s.history[-8:]
    msgs.append({"role":"user","content":user})
    return msgs

# ========= OpenAI call =========
def call_openai(messages: List[Dict[str,str]], stream: bool) -> Generator[str,None,str] | str:
    if not (_HAS_OPENAI and OPENAI_API_KEY):
        txt = "⚠️ OpenAI disabled on server; running in dev mode."
        if stream:
            def _g():  # type: ignore
                yield txt
            return _g()
        return txt
    client = OpenAI(api_key=OPENAI_API_KEY)
    if stream:
        resp = client.chat.completions.create(model=OPENAI_MODEL, messages=messages, temperature=0.5, stream=True)
        def _gen():  # type: ignore
            for ev in resp:
                chunk = ev.choices[0].delta.content or ""
                if chunk: yield chunk
        return _gen()
    else:
        resp = client.chat.completions.create(model=OPENAI_MODEL, messages=messages, temperature=0.5)
        return resp.choices[0].message.content or ""

# ========= App + CORS =========
ALLOWED_ORIGINS = {
    "https://dturnover.github.io",
    # add locals if needed:
    "http://localhost:3000", "http://127.0.0.1:3000",
    "http://localhost:5500", "http://127.0.0.1:5500",
}

app = FastAPI()
# Base CORS (we still echo origin below so cookies work)
app.add_middleware(
    CORSMiddleware,
    allow_origins=list(ALLOWED_ORIGINS),
    allow_credentials=True,
    allow_methods=["GET","POST","OPTIONS"],
    allow_headers=["*"],
    expose_headers=["*"],
    max_age=600,
)

# Echo Origin (some proxies strip framework CORS; this forces the header)
@app.middleware("http")
async def echo_origin_header(request: Request, call_next):
    origin = request.headers.get("origin", "")
    resp: Response = await call_next(request)
    if origin in ALLOWED_ORIGINS:
        resp.headers["Access-Control-Allow-Origin"] = origin
        resp.headers["Vary"] = "Origin, Accept-Encoding"
        resp.headers["Access-Control-Allow-Credentials"] = "true"
        resp.headers["Access-Control-Allow-Methods"] = "GET,POST,OPTIONS"
        resp.headers["Access-Control-Allow-Headers"] = "*"
        resp.headers["Access-Control-Expose-Headers"] = "*"
    return resp

# ========= Utilities =========
def sse_event(event: Optional[str], data: str) -> str:
    if event:
        return f"event: {event}\n" + "\n".join(f"data: {line}" for line in data.splitlines()) + "\n\n"
    return "\n".join(f"data: {line}" for line in data.splitlines()) + "\n\n"

# ========= Routes =========
@app.get("/")
def root():
    return PlainTextResponse("ok")

@app.get("/health")
def health():
    return PlainTextResponse("OK")

@app.get("/diag_rag")
def diag():
    return {"ok": True, "corpora": diag_rag_counts()}

def _json_response(resp_text: str, sources: List[Dict[str,Any]], corpus: Optional[str], remembered: Optional[Dict[str,str]], base: Response) -> JSONResponse:
    jr = JSONResponse({"response": resp_text, "sources": sources, "selected_corpus": corpus, "remembered": remembered})
    # forward cookie header from get_or_create_sid response
    for k, v in base.headers.items():
        jr.headers[k] = v
    return jr

@app.post("/chat")
async def chat(request: Request):
    body = await request.json()
    message = (body.get("message") or "").strip()

    base = Response()
    _sid, s = get_or_create_sid(request, base)
    remembered = parse_and_remember(message, s)
    try_set_faith(message, s)

    # 1) memory fast-path
    mem = lookup_memory_answer(message, s)
    if mem:
        s.history.append({"role":"user","content":message})
        s.history.append({"role":"assistant","content":mem})
        return _json_response(mem, [], None, remembered, base)

    # 2) proactive scripture (faith known + emotional) OR explicit scripture ask
    if s.faith and (is_emotional(message) or wants_scripture(message)):
        corpus = detect_corpus(message, s)
        sources, verse = search_scripture(message, corpus)
        if verse:
            s.history.append({"role":"user","content":message})
            s.history.append({"role":"assistant","content":verse})
            return _json_response(verse, sources, corpus, remembered, base)

    # 3) normal GPT
    out = call_openai(_build_messages(s, message), stream=False)
    reply = out if isinstance(out, str) else "".join(list(out))
    s.history.append({"role":"user","content":message})
    s.history.append({"role":"assistant","content":reply})
    return _json_response(reply, [], s.faith, remembered, base)

@app.post("/chat_sse")
async def chat_sse(request: Request):
    body = await request.json()
    message = (body.get("message") or "").strip()

    base = Response()
    _sid, s = get_or_create_sid(request, base)
    remembered = parse_and_remember(message, s)
    try_set_faith(message, s)

    def gen() -> Generator[str, None, None]:
        # establish stream
        yield ": connected\n\n"

        # memory?
        mem = lookup_memory_answer(message, s)
        if mem:
            s.history.append({"role":"user","content":message})
            s.history.append({"role":"assistant","content":mem})
            for chunk in re.findall(r".{1,220}(?:\s+|$)", mem):
                yield sse_event(None, chunk.strip())
            meta = json.dumps({"sources": [], "selected_corpus": s.faith, "remembered": remembered})
            yield sse_event("sources", meta)
            return

        # scripture?
        if s.faith and (is_emotional(message) or wants_scripture(message)):
            corpus = detect_corpus(message, s)
            sources, verse = search_scripture(message, corpus)
            if verse:
                s.history.append({"role":"user","content":message})
                s.history.append({"role":"assistant","content":verse})
                for chunk in re.findall(r".{1,220}(?:\s+|$)", verse):
                    yield sse_event(None, chunk.strip())
                meta = json.dumps({"sources": sources, "selected_corpus": corpus, "remembered": remembered})
                yield sse_event("sources", meta)
                return

        # GPT stream
        stream = call_openai(_build_messages(s, message), stream=True)  # type: ignore
        buf = ""
        for piece in stream:  # generator of text fragments
            buf += piece
            if re.search(r"[.!?]\s+$", buf) or len(buf) >= 140:
                yield sse_event(None, buf)
                buf = ""
        if buf:
            yield sse_event(None, buf)

        s.history.append({"role":"user","content":message})
        # trailing meta
        meta = json.dumps({"sources": [], "selected_corpus": s.faith, "remembered": remembered})
        yield sse_event("sources", meta)

    headers = dict(base.headers)
    return StreamingResponse(gen(), media_type="text/event-stream", headers=headers)

# Optional: GET streaming (good for quick curl tests)
@app.get("/chat_sse_get")
def chat_sse_get(request: Request, q: str):
    base = Response()
    _sid, s = get_or_create_sid(request, base)
    remembered = parse_and_remember(q, s)
    try_set_faith(q, s)

    def gen() -> Generator[str,None,None]:
        yield ": connected\n\n"
        if s.faith and (is_emotional(q) or wants_scripture(q)):
            corpus = detect_corpus(q, s)
            sources, verse = search_scripture(q, corpus)
            if verse:
                for chunk in re.findall(r".{1,220}(?:\s+|$)", verse):
                    yield sse_event(None, chunk.strip())
                yield sse_event("sources", json.dumps({"sources": sources, "selected_corpus": corpus, "remembered": remembered}))
                return
        for chunk in ["This is a GET stream endpoint.\n"]:
            yield sse_event(None, chunk)
        yield sse_event("sources", json.dumps({"sources": [], "selected_corpus": s.faith, "remembered": remembered}))

    headers = dict(base.headers)
    return StreamingResponse(gen(), media_type="text/event-stream", headers=headers)
