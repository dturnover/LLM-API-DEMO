# app.py — Phases 0–5: health, non-stream, SSE, memory + faith (no cookies)
import os, json, asyncio, re, uuid
from typing import Optional, Dict, Any

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse, JSONResponse, StreamingResponse

# --- OpenAI (non-fatal if missing) ---
try:
    from openai import OpenAI  # pip install openai>=1.0
    _HAS_OPENAI = True
except Exception:
    OpenAI = None  # type: ignore
    _HAS_OPENAI = False

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL   = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

app = FastAPI()

# CORS: allow Pages + local dev; no credentials/cookies used
ALLOWED_ORIGINS = {
    "https://dturnover.github.io",
    "http://localhost:3000", "http://127.0.0.1:3000",
    "http://localhost:5500", "http://127.0.0.1:5500",
}
app.add_middleware(
    CORSMiddleware,
    allow_origins=list(ALLOWED_ORIGINS),
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
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
def _openai_nonstream(prompt: str) -> str:
    if not (_HAS_OPENAI and OPENAI_API_KEY):
        return f"DEV: received \"{prompt}\""
    client = OpenAI(api_key=OPENAI_API_KEY)
    resp = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.5,
    )
    return resp.choices[0].message.content or ""

# --------------- Phase 5: memory/faith -----------
class Session:
    def __init__(self) -> None:
        self.facts: Dict[str, str] = {}
        self.faith: Optional[str] = None  # "bible"|"quran"|"sutta_pitaka"

SESSIONS: Dict[str, Session] = {}

def _get_sid(request: Request) -> str:
    sid = request.headers.get("x-session-id") or request.query_params.get("sid") or ""
    if not sid:
        sid = uuid.uuid4().hex
    if sid not in SESSIONS:
        SESSIONS[sid] = Session()
    return sid

REMEMBER_RE = re.compile(r"\bremember\b", re.I)
ASSIGN_RE   = re.compile(r"^\s*(?P<k>.+?)\s+(?:is|are|=)\s+(?P<v>.+?)\s*$", re.I)
WHAT_COLOR  = re.compile(r"^\s*what\s+(?:color|colour)\s+is\s+(?P<x>.+?)\??\s*$", re.I)
WHAT_IS     = re.compile(r"^\s*what\s+is\s+(?P<x>.+?)\??\s*$", re.I)

FAITH_MAP = {
    "christian": "bible", "christianity":"bible", "bible":"bible",
    "muslim":"quran", "islam":"quran", "qur":"quran", "quran":"quran",
    "buddhist":"sutta_pitaka", "buddhism":"sutta_pitaka", "sutta":"sutta_pitaka",
}

def _normalize_key(s: str) -> str:
    s = re.sub(r"^(the|a|an|this|that)\s+", "", s.strip(), flags=re.I)
    return re.sub(r"\s+", " ", s).lower()

def _try_remember(msg: str, sess: Session) -> Optional[str]:
    if not REMEMBER_RE.search(msg): return None
    body = re.sub(r"^\s*remember\s*(that|:)?\s*", "", msg, flags=re.I).strip()
    m = ASSIGN_RE.match(body)
    if m:
        k = _normalize_key(m.group("k")); v = m.group("v").strip()
        if k: sess.facts[k] = v; return f"Noted: {k} = {v}"
    # fallback: first 4 words as key
    parts = body.split()
    if parts:
        k = _normalize_key(" ".join(parts[:4])); v = " ".join(parts[4:]) or body
        sess.facts[k] = v; return f"Noted: {k} = {v}"
    return "Noted."

def _lookup_memory(msg: str, sess: Session) -> Optional[str]:
    m = WHAT_COLOR.match(msg) or WHAT_IS.match(msg)
    if not m: return None
    k = _normalize_key(m.group("x"))
    return sess.facts.get(k) or sess.facts.get(k.replace("the ", "", 1))

def _try_set_faith(msg: str, sess: Session) -> Optional[str]:
    low = msg.lower()
    for k, v in FAITH_MAP.items():
        if re.search(rf"\b{re.escape(k)}\b", low):
            prev = sess.faith
            sess.faith = v
            if prev != v:
                return f"Faith set to {v}"
            break
    return None

# ---------------- Phase 1–2: /chat ----------------
@app.post("/chat")
async def chat(request: Request):
    try:
        body = await request.json()
    except Exception:
        body = {}
    msg = (body.get("message") or "").strip()

    sid = _get_sid(request)
    sess = SESSIONS[sid]

    # Phase 1: ping
    if msg.lower() == "ping":
        return JSONResponse({"response": "pong", "sid": sid}, headers={"X-Session-Id": sid})

    # Phase 5: remember / recall / faith
    remembered = _try_remember(msg, sess)
    if remembered:
        return JSONResponse({"response": remembered, "sid": sid}, headers={"X-Session-Id": sid})

    recall = _lookup_memory(msg, sess)
    if recall:
        return JSONResponse({"response": recall, "sid": sid}, headers={"X-Session-Id": sid})

    faith_ack = _try_set_faith(msg, sess)
    if faith_ack:
        return JSONResponse({"response": faith_ack, "sid": sid}, headers={"X-Session-Id": sid})

    # Phase 2: OpenAI non-stream fallback
    text = _openai_nonstream(msg)
    return JSONResponse({"response": text, "sid": sid}, headers={"X-Session-Id": sid})

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

# ----------- Phase 3–4: /chat_sse ----------------
@app.api_route("/chat_sse", methods=["GET", "POST"])
async def chat_sse(request: Request, q: Optional[str] = None, sid: Optional[str] = None):
    # Note: Phase 5 state is kept per X-Session-Id or ?sid=... if provided,
    # but SSE itself does not yet use memory/faith logic.
    if request.method == "POST":
        try:
            body = await request.json()
        except Exception:
            body = {}
        message = (body.get("message") or "").strip()
    else:
        message = (q or "").strip()

    # adopt sid if given for future phases
    if sid:
        if sid not in SESSIONS:
            SESSIONS[sid] = Session()

    async def agen():
        yield ": connected\n\n"
        yield _sse_event("ping", "hi")

        # Transport-only probe
        if message.lower() in {"abc", "transport", "probe"}:
            for tok in ["A", "B", "C", "D", "E"]:
                yield _sse_data({"text": tok})
                await asyncio.sleep(0.15)
            yield _sse_event("done", {})
            return

        # OpenAI streaming (or dev stream)
        if not (_HAS_OPENAI and OPENAI_API_KEY):
            for tok in ["dev", " ", "stream", " ", "ok"]:
                yield _sse_data({"text": tok})
                await asyncio.sleep(0.12)
            yield _sse_event("done", {})
            return

        client = OpenAI(api_key=OPENAI_API_KEY)
        stream = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role": "user", "content": message}],
            temperature=0.5,
            stream=True,
        )
        for ev in stream:
            delta = ev.choices[0].delta.content or ""
            if delta:
                yield _sse_data({"text": delta})
                await asyncio.sleep(0)
        yield _sse_event("done", {})

    headers = {
        "Cache-Control": "no-cache",
        "X-Accel-Buffering": "no",
    }
    return StreamingResponse(agen(), media_type="text/event-stream", headers=headers)
