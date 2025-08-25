# app.py — Phases 0–6: health, non-stream, SSE, dynamic memory+faith (no cookies)
import os, json, asyncio, re, uuid
from typing import Optional, Dict, Any, List

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse, JSONResponse, StreamingResponse

# --- OpenAI (safe if missing) ---
try:
    from openai import OpenAI  # pip install openai>=1.0
    _HAS_OPENAI = True
except Exception:
    OpenAI = None  # type: ignore
    _HAS_OPENAI = False

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL   = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

app = FastAPI()

# CORS (no credentials/cookies)
ALLOWED_ORIGINS = {
    "https://dturnover.github.io",
    "http://localhost:3000", "http://127.0.0.1:3000",
    "http://localhost:5500", "http://127.0.0.1:5500",
}
app.add_middleware(
    CORSMiddleware,
    allow_origins=list(ALLOWED_ORIGINS),
    allow_credentials=False,
    allow_methods=["GET","POST","OPTIONS"],
    allow_headers=["*"],
    expose_headers=["*"],
    max_age=600,
)

# ---------------- Phase 0: health ----------------
@app.get("/health")
def health():
    return PlainTextResponse("OK")

@app.get("/")
def root():
    return PlainTextResponse("ok")

# ---------------- OpenAI helpers -----------------
def _openai_nonstream(prompt: str, system: Optional[str] = None, history: Optional[List[Dict[str,str]]] = None) -> str:
    if not (_HAS_OPENAI and OPENAI_API_KEY):
        return f"DEV: received \"{prompt}\""
    client = OpenAI(api_key=OPENAI_API_KEY)
    messages: List[Dict[str,str]] = []
    if system:
        messages.append({"role":"system","content":system})
    if history:
        messages.extend(history[-8:])
    messages.append({"role":"user","content":prompt})
    resp = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=messages,
        temperature=0.4,
    )
    return resp.choices[0].message.content or ""

def _openai_stream(prompt: str, system: Optional[str], history: Optional[List[Dict[str,str]]]):
    if not (_HAS_OPENAI and OPENAI_API_KEY):
        async def _dev():
            for tok in ["dev"," ","stream"," ","ok"]:
                yield {"text": tok}
                await asyncio.sleep(0.12)
        return _dev()

    client = OpenAI(api_key=OPENAI_API_KEY)
    messages: List[Dict[str,str]] = []
    if system:
        messages.append({"role":"system","content":system})
    if history:
        messages.extend(history[-8:])
    messages.append({"role":"user","content":prompt})

    def _gen_sync():
        stream = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=messages,
            temperature=0.4,
            stream=True,
        )
        for ev in stream:
            delta = ev.choices[0].delta.content or ""
            if delta:
                yield {"text": delta}
    # wrap sync generator into async for StreamingResponse
    async def _agen():
        for item in _gen_sync():
            yield item
            await asyncio.sleep(0)
    return _agen()

# ---------------- Session + memory ----------------
class Session:
    def __init__(self) -> None:
        self.facts: Dict[str, str] = {}
        self.faith: Optional[str] = None  # "bible"|"quran"|"sutta_pitaka"
        self.name: Optional[str] = None
        self.prefs: Dict[str, Any] = {}
        self.history: List[Dict[str,str]] = []  # rolling chat turns

SESSIONS: Dict[str, Session] = {}

def _get_sid(request: Request) -> str:
    sid = request.headers.get("x-session-id") or request.query_params.get("sid") or ""
    if not sid:
        sid = uuid.uuid4().hex
    if sid not in SESSIONS:
        SESSIONS[sid] = Session()
    return sid

# --- memory parsing ---
REMEMBER_RE = re.compile(r"\bremember\b", re.I)
ASSIGN_RE   = re.compile(r"^\s*(?P<k>.+?)\s+(?:is|are|=)\s+(?P<v>.+?)\s*$", re.I)
WHAT_COLOR  = re.compile(r"^\s*what\s+(?:color|colour)\s+is\s+(?P<x>.+?)\??\s*$", re.I)
WHAT_IS     = re.compile(r"^\s*what\s+is\s+(?P<x>.+?)\??\s*$", re.I)

# auto-facts
NAME_PATS = [
    re.compile(r"\b(?:i\s*am|i'm|im)\s+(?P<name>[a-z][a-z'-]*(?:\s+[a-z][a-z'-]*){0,2})\b", re.I),
    re.compile(r"\bmy\s+name\s+is\s+(?P<name>[a-z][a-z'-]*(?:\s+[a-z][a-z'-]*){0,2})\b", re.I),
    re.compile(r"\bcall\s+me\s+(?P<name>[a-z][a-z'-]*)\b", re.I),
]

FAITH_MAP = {
    "christian": "bible", "christianity":"bible", "bible":"bible",
    "muslim":"quran", "islam":"quran", "qur":"quran", "quran":"quran",
    "buddhist":"sutta_pitaka", "buddhism":"sutta_pitaka", "sutta":"sutta_pitaka",
}
FAITH_PAT = re.compile(r"\b(i\s*am|i'm|im)\s+(?P<faith>christian|muslim|islam|buddhist|buddhism)\b", re.I)

def _normalize_key(s: str) -> str:
    s = re.sub(r"^(the|a|an|this|that|my|our)\s+", "", s.strip(), flags=re.I)
    return re.sub(r"\s+", " ", s).lower()

def _try_remember_line(msg: str, sess: Session) -> Optional[str]:
    if not REMEMBER_RE.search(msg): return None
    body = re.sub(r"^\s*remember\s*(that|:)?\s*", "", msg, flags=re.I).strip()
    m = ASSIGN_RE.match(body)
    if m:
        k = _normalize_key(m.group("k")); v = m.group("v").strip()
        if k: sess.facts[k] = v; return f"Noted: {k} = {v}"
    # fallback: "remember X: Y"
    if ":" in body:
        k, v = body.split(":", 1)
        k = _normalize_key(k); v = v.strip()
        if k: sess.facts[k] = v; return f"Noted: {k} = {v}"
    # minimal fallback
    if body:
        k = _normalize_key(body.split()[0]); v = body
        sess.facts[k] = v; return f"Noted: {k} = {v}"
    return "Noted."

def _auto_facts(msg: str, sess: Session) -> Optional[str]:
    low = msg.lower()

    # name
    for p in NAME_PATS:
        m = p.search(low)
        if m:
            name = m.group("name").strip()
            # avoid capturing trailing punctuation
            name = re.sub(r"[^\w\s'-].*$", "", name).strip()
            if name:
                pretty = " ".join(w.capitalize() for w in name.split())
                sess.name = pretty
                sess.facts["name"] = pretty
                return f"Noted: name = {pretty}"

    # faith (also mapped)
    m = FAITH_PAT.search(low)
    if m:
        for k, v in FAITH_MAP.items():
            if re.search(rf"\b{re.escape(k)}\b", low):
                prev = sess.faith
                sess.faith = v
                if prev != v:
                    sess.facts["faith"] = v
                    return f"Faith set to {v}"
                break
    # lightweight prefs (example)
    if "short answers" in low or "be concise" in low:
        sess.prefs["concise"] = True
        return "Noted: preference = concise"

    return None

def _lookup_memory(msg: str, sess: Session) -> Optional[str]:
    low = msg.lower()

    # natural recalls
    if re.search(r"\bwhat('?s| is)\s+my\s+name\b|\bwho\s+am\s+i\b", low):
        return sess.name or sess.facts.get("name")
    if re.search(r"\bwhat('?s| is)\s+my\s+(faith|religion)\b", low):
        return sess.faith or sess.facts.get("faith")

    m = WHAT_COLOR.match(msg) or WHAT_IS.match(msg)
    if m:
        k = _normalize_key(m.group("x"))
        return sess.facts.get(k) or sess.facts.get(k.replace("the ","",1))
    return None

def _system_prompt(sess: Session) -> str:
    lines = [
        "You are Fight Chaplain: calm, concise, encouraging.",
        "Offer practical corner-coach guidance with spiritual grounding.",
        "Keep answers tight; avoid filler.",
    ]
    if sess.faith:
        lines.append(f"User faith: {sess.faith}. Prefer that tradition when offering encouragement.")
    if sess.name:
        lines.append(f"User goes by: {sess.name}.")
    if sess.prefs.get("concise"):
        lines.append("Be extra concise.")
    if sess.facts:
        lines.append("STICKY_NOTES:")
        for k, v in list(sess.facts.items())[:24]:
            lines.append(f"- {k} = {v}")
    return "\n".join(lines)

def _record_turn(sess: Session, role: str, content: str) -> None:
    sess.history.append({"role": role, "content": content})
    if len(sess.history) > 12:
        del sess.history[:-12]

# ---------------- /chat (non-stream) ----------------
@app.post("/chat")
async def chat(request: Request):
    try:
        body = await request.json()
    except Exception:
        body = {}
    msg = (body.get("message") or "").strip()

    sid = _get_sid(request)
    sess = SESSIONS[sid]

    if msg.lower() == "ping":
        return JSONResponse({"response":"pong", "sid": sid}, headers={"X-Session-Id": sid})

    # memory ops
    remembered = _try_remember_line(msg, sess)
    if remembered:
        _record_turn(sess, "user", msg); _record_turn(sess, "assistant", remembered)
        return JSONResponse({"response": remembered, "sid": sid}, headers={"X-Session-Id": sid})

    auto = _auto_facts(msg, sess)
    if auto:
        _record_turn(sess, "user", msg); _record_turn(sess, "assistant", auto)
        return JSONResponse({"response": auto, "sid": sid}, headers={"X-Session-Id": sid})

    recalled = _lookup_memory(msg, sess)
    if recalled:
        _record_turn(sess, "user", msg); _record_turn(sess, "assistant", recalled)
        return JSONResponse({"response": recalled, "sid": sid}, headers={"X-Session-Id": sid})

    # OpenAI (with memory + history)
    reply = _openai_nonstream(msg, system=_system_prompt(sess), history=sess.history)
    _record_turn(sess, "user", msg); _record_turn(sess, "assistant", reply)
    return JSONResponse({"response": reply, "sid": sid}, headers={"X-Session-Id": sid})

# ---------------- SSE helpers --------------------
def _sse_data(obj: Dict[str, Any] | str) -> str:
    if isinstance(obj, dict):
        return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n"
    return "".join([f"data: {ln}\n" for ln in str(obj).splitlines()]) + "\n"

def _sse_event(event: str, data: Dict[str, Any] | str = "") -> str:
    out = f"event: {event}\n"
    if data != "":
        if isinstance(data, dict):
            out += f"data: {json.dumps(data, ensure_ascii=False)}\n"
        else:
            for ln in str(data).splitlines():
                out += f"data: {ln}\n"
    return out + "\n"

# --------------- /chat_sse (stream with memory) ---------------
@app.api_route("/chat_sse", methods=["GET","POST"])
async def chat_sse(request: Request, q: Optional[str] = None, sid: Optional[str] = None):
    # sid comes from header or query
    header_sid = request.headers.get("x-session-id") or ""
    sid = sid or header_sid
    if not sid:
        sid = _get_sid(request)  # create one if missing
    if sid not in SESSIONS:
        SESSIONS[sid] = Session()
    sess = SESSIONS[sid]

    if request.method == "POST":
        try:
            body = await request.json()
        except Exception:
            body = {}
        message = (body.get("message") or "").strip()
    else:
        message = (q or "").strip()

    async def agen():
        yield ": connected\n\n"
        yield _sse_event("ping", "hi")

        # memory ops first
        remembered = _try_remember_line(message, sess)
        if remembered:
            _record_turn(sess, "user", message)
            # stream remembered text in chunks
            for chunk in re.findall(r".{1,220}(?:\s+|$)", remembered):
                yield _sse_data({"text": chunk.strip()})
                await asyncio.sleep(0)
            _record_turn(sess, "assistant", remembered)
            yield _sse_event("done", {"sid": sid})
            return

        auto = _auto_facts(message, sess)
        if auto:
            _record_turn(sess, "user", message)
            for chunk in re.findall(r".{1,220}(?:\s+|$)", auto):
                yield _sse_data({"text": chunk.strip()})
                await asyncio.sleep(0)
            _record_turn(sess, "assistant", auto)
            yield _sse_event("done", {"sid": sid})
            return

        recalled = _lookup_memory(message, sess)
        if recalled:
            _record_turn(sess, "user", message)
            for chunk in re.findall(r".{1,220}(?:\s+|$)", recalled):
                yield _sse_data({"text": chunk.strip()})
                await asyncio.sleep(0)
            _record_turn(sess, "assistant", recalled)
            yield _sse_event("done", {"sid": sid})
            return

        # OpenAI streaming with memory + history
        streamer = _openai_stream(message, system=_system_prompt(sess), history=sess.history)
        _record_turn(sess, "user", message)

        async for piece in streamer:
            yield _sse_data(piece)   # {"text": "..."}
        # not adding model reply to history here (token merge); keep simple
        yield _sse_event("done", {"sid": sid})

    headers = {
        "Cache-Control": "no-cache",
        "X-Accel-Buffering": "no",
        "X-Session-Id": sid,
    }
    return StreamingResponse(agen(), media_type="text/event-stream", headers=headers)
