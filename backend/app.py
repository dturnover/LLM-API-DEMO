# app.py — Phases 0–4 (health, non-stream, transport-only SSE, OpenAI SSE)
import os, json, asyncio
from typing import Optional, Iterable

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

# CORS: allow your Pages origin + local dev
ALLOWED_ORIGINS = {
    "https://dturnover.github.io",
    "http://localhost:3000", "http://127.0.0.1:3000",
    "http://localhost:5500", "http://127.0.0.1:5500",
}
app.add_middleware(
    CORSMiddleware,
    allow_origins=list(ALLOWED_ORIGINS),
    allow_credentials=False,                 # we’re not using cookies in Phases 0–4
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["*"],
    max_age=600,
)

# ---------- Phase 0: health ----------
@app.get("/health")
def health():
    return PlainTextResponse("OK")

@app.get("/")
def root():
    return PlainTextResponse("ok")

# ---------- Phase 1–2: /chat (non-stream, then OpenAI non-stream) ----------
def _openai_nonstream(prompt: str) -> str:
    if not (_HAS_OPENAI and OPENAI_API_KEY):
        # Dev stub when no key/lib present
        return f"DEV: received \"{prompt}\""
    client = OpenAI(api_key=OPENAI_API_KEY)
    resp = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.5,
    )
    return resp.choices[0].message.content or ""

@app.post("/chat")
async def chat(request: Request):
    try:
        body = await request.json()
    except Exception:
        body = {}
    msg = (body.get("message") or "").strip()
    if msg.lower() == "ping":
        return JSONResponse({"response": "pong"})
    text = _openai_nonstream(msg)
    return JSONResponse({"response": text})

# ---------- helpers for SSE ----------
def sse_data(obj: dict | str) -> str:
    if isinstance(obj, dict):
        payload = json.dumps(obj, ensure_ascii=False)
        return f"data: {payload}\n\n"
    # string payload
    lines = str(obj).splitlines() or [""]
    return "".join([f"data: {ln}\n" for ln in lines]) + "\n"

def sse_event(event: str, data: dict | str = "") -> str:
    out = f"event: {event}\n"
    if data != "":
        if isinstance(data, dict):
            out += f"data: {json.dumps(data, ensure_ascii=False)}\n"
        else:
            for ln in str(data).splitlines():
                out += f"data: {ln}\n"
    return out + "\n"

# ---------- Phase 3–4: /chat_sse (transport-only + OpenAI streaming) ----------
@app.api_route("/chat_sse", methods=["GET", "POST"])
async def chat_sse(request: Request, q: Optional[str] = None):
    # Determine message (GET uses ?q=..., POST uses {"message":...})
    if request.method == "POST":
        try:
            body = await request.json()
        except Exception:
            body = {}
        message = (body.get("message") or "").strip()
    else:
        message = (q or "").strip()

    async def agen():
        # Immediate heartbeat so proxies don’t 0-length the response
        yield ": connected\n\n"
        # Disable idle buffering with periodic ping
        yield sse_event("ping", "hi")

        # ---- Phase 3: transport-only stream (no OpenAI) ----
        if message.lower() in {"abc", "transport", "probe"}:
            for tok in ["A", "B", "C", "D", "E"]:
                yield sse_data({"text": tok})
                await asyncio.sleep(0.15)
            yield sse_event("done", {})
            return

        # ---- Phase 4: OpenAI streaming (fallback to dev stream if no key) ----
        if not (_HAS_OPENAI and OPENAI_API_KEY):
            # Dev token stream
            for tok in ["dev", " ", "stream", " ", "ok"]:
                yield sse_data({"text": tok})
                await asyncio.sleep(0.12)
            yield sse_event("done", {})
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
                yield sse_data({"text": delta})
                await asyncio.sleep(0)  # let loop flush
        yield sse_event("done", {})

    headers = {
        "Cache-Control": "no-cache",
        "X-Accel-Buffering": "no",   # ask proxies not to buffer
    }
    return StreamingResponse(agen(), media_type="text/event-stream", headers=headers)
