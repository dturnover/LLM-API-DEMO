# app.py
import os
import re
import json
import uuid
import time
from typing import Dict, List, Optional, Generator, Any, Tuple
from dataclasses import dataclass, field

from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

# === OpenAI client (works with openai>=1.0) ===
try:
    from openai import OpenAI
    _HAS_OPENAI = True
except Exception:
    _HAS_OPENAI = False
    OpenAI = None  # type: ignore

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")  # pick any chat model you use
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# ---------- Simple in-memory session store ----------
# NOTE: this resets on process restart (Render free dyno etc.). Swap to Redis for durability.
@dataclass
class SessionState:
    faith: Optional[str] = None               # 'bible' | 'quran' | 'sutta_pitaka'
    facts: Dict[str, str] = field(default_factory=dict)  # "the rose" -> "red"
    summary: str = ""                         # conversation summary (optional future use)
    history: List[Dict[str, str]] = field(default_factory=list)  # recent turns for model
    last_object: Optional[str] = None         # naive "this/that X" anchor

SESSIONS: Dict[str, SessionState] = {}

def get_or_create_sid(request: Request, response: Response) -> Tuple[str, SessionState]:
    sid = request.cookies.get("sid")
    if not sid:
        sid = uuid.uuid4().hex
        response.set_cookie("sid", sid, httponly=True, samesite="lax", path="/")
    if sid not in SESSIONS:
        SESSIONS[sid] = SessionState()
    return sid, SESSIONS[sid]

# ---------- Tiny RAG (fallback) ----------
# If you already have an index in your project, feel free to replace the
# functions below with your existing search utilities. The interface here is:
#   search_scripture(query, corpus) -> (sources_list, best_snippet)
RAG_ROOT = os.path.join(os.path.dirname(__file__), "rag_data")
CORPUS_FILES = {
    "bible": "bible.jsonl",
    "quran": "quran.jsonl",
    "sutta_pitaka": "sutta_pitaka.jsonl",
}

_LOADED_CORPORA: Dict[str, List[Dict[str, Any]]] = {}

def _load_corpus(corpus: str) -> List[Dict[str, Any]]:
    if corpus not in CORPUS_FILES:
        return []
    if corpus in _LOADED_CORPORA:
        return _LOADED_CORPORA[corpus]
    path = os.path.join(RAG_ROOT, CORPUS_FILES[corpus])
    docs: List[Dict[str, Any]] = []
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                try:
                    obj = json.loads(line)
                    # normalize expected fields
                    obj.setdefault("text", obj.get("content") or obj.get("body") or "")
                    obj.setdefault("page", obj.get("page") or obj.get("p") or i)
                    obj["id"] = obj.get("id") or f"p{obj['page']}_c{i}"
                    obj["corpus"] = corpus
                    docs.append(obj)
                except Exception:
                    continue
    _LOADED_CORPORA[corpus] = docs
    return docs

def diag_rag_counts() -> Dict[str, int]:
    out = {}
    for c in CORPUS_FILES:
        out[c] = len(_load_corpus(c))
    return out

def _score_hit(q_terms: List[str], text: str) -> int:
    # super-naive keyword score; good enough for short verse pulls
    t = text.lower()
    score = 0
    for w in q_terms:
        if not w: continue
        score += t.count(w)
    return score

def search_scripture(query: str, corpus: str, k: int = 5) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    docs = _load_corpus(corpus)
    if not docs:
        return [], None
    q = re.sub(r"[^a-z0-9\s]", " ", query.lower())
    q_terms = [w for w in q.split() if len(w) > 2]
    scored = []
    for d in docs:
        s = _score_hit(q_terms, d.get("text", ""))
        if s > 0:
            scored.append((s, d))
    if not scored:
        # fallback: just return a generic encouraging line if nothing matched
        return [], None
    scored.sort(key=lambda x: x[0], reverse=True)
    top = [d for _, d in scored[:k]]
    # pick first as best snippet
    best = top[0].get("text", "")
    return top, best

# ---------- Memory helpers ----------
REMEMBER_PAT = re.compile(
    r"^\s*(?:remember\s*(?:that|:)?\s*)?(?P<body>.+)$",
    re.IGNORECASE
)

IS_PATTERN = re.compile(
    r"^\s*(?P<key>.+?)\s+(?:is|are|=)\s+(?P<val>.+?)\s*$",
    re.IGNORECASE
)

WHAT_COLOR_PAT = re.compile(
    r"^\s*what\s+(?:color|colour)\s+is\s+(?:the\s+)?(?P<thing>[^?!.]+)\??\s*$",
    re.IGNORECASE
)

WHAT_IS_PAT = re.compile(
    r"^\s*what\s+is\s+(?:the\s+)?(?P<thing>[^?!.]+)\??\s*$",
    re.IGNORECASE
)

FAITH_MAP = {
    "christian": "bible",
    "christianity": "bible",
    "muslim": "quran",
    "islam": "quran",
    "buddhist": "sutta_pitaka",
    "buddhism": "sutta_pitaka",
}

def normalize_key(s: str) -> str:
    s = s.strip().lower()
    # drop leading 'the', 'a', 'this', 'that', basic punctuation
    s = re.sub(r"^(the|a|an|this|that)\s+", "", s)
    s = re.sub(r"\s+", " ", s)
    return s

def try_set_faith(message: str, sess: SessionState) -> Optional[str]:
    m = message.lower()
    for k, v in FAITH_MAP.items():
        if re.search(rf"\b{i_escape(k)}\b", m):
            sess.faith = v
            return v
    return None

def i_escape(s: str) -> str:
    return re.escape(s)

def parse_and_remember(message: str, sess: SessionState) -> Optional[Dict[str, str]]:
    """
    Supports:
      - 'remember that X is Y'
      - 'remember: my dog tucker died 7 years ago'
      - 'remember this: the strawberry is black'
    """
    text = message.strip()
    if not re.search(r"\bremember\b", text, re.IGNORECASE):
        return None

    m = REMEMBER_PAT.match(text)
    if not m:
        return None

    body = m.group("body").strip()

    # Try to anchor 'this/that <noun>' to last object if present (very naive)
    if body.lower().startswith(("this ", "that ")) and sess.last_object:
        body = body.lower().replace("this", sess.last_object, 1).replace("that", sess.last_object, 1)

    # Try X is Y
    m2 = IS_PATTERN.match(body)
    if m2:
        key = normalize_key(m2.group("key"))
        val = m2.group("val").strip()
        if key:
            sess.facts[key] = val
            sess.last_object = key
            return {"key": key, "value": val}

    # Otherwise, store the whole clause, choose a simple key
    # e.g., 'my dog tucker died 7 years ago' -> key 'my dog tucker', value 'died 7 years ago'
    parts = re.split(r"\bis\b|=|,|;|—|-", body, maxsplit=1, flags=re.IGNORECASE)
    if len(parts) == 2:
        key, val = parts[0].strip(), parts[1].strip()
    else:
        # fallback: first 4 words = key, rest = value
        words = body.split()
        key = " ".join(words[:4])
        val = " ".join(words[4:]) or body
    key_n = normalize_key(key)
    sess.facts[key_n] = val
    sess.last_object = key_n
    return {"key": key_n, "value": val}

def lookup_memory_answer(message: str, sess: SessionState) -> Optional[str]:
    for pat in (WHAT_COLOR_PAT, WHAT_IS_PAT):
        m = pat.match(message)
        if m:
            thing_raw = m.group("thing").strip()
            thing = normalize_key(thing_raw)
            # direct
            if thing in sess.facts:
                return sess.facts[thing]
            # try 'the ' prefix or last_object
            if thing.startswith("the "):
                k = thing[4:]
                if k in sess.facts:
                    return sess.facts[k]
            # If user says 'this X' and we saved 'X'
            if thing.startswith("this "):
                k = thing.replace("this ", "", 1)
                k = normalize_key(k)
                if k in sess.facts:
                    return sess.facts[k]
    return None

# ---------- Prompt + model ----------
def build_system_prompt(sess: SessionState) -> str:
    lines = [
        "You are 'Fight Chaplain'—a calm, practical corner coach and chaplain.",
        "Be concise, plain, and supportive. Avoid faux-Elizabethan flourish.",
        "Use scripture only when asked or when user implies they'd like a verse.",
        "When scripture is requested, quote verbatim, then add a one-line takeaway.",
        "If a verse is from a corpus, append a bracket citation like [S1].",
        "If the user asks about a previously remembered fact, answer directly and briefly.",
    ]
    # Faith hint
    if sess.faith:
        lines.append(f"User faith: {sess.faith}. Prefer this corpus for verses.")

    # Sticky notes
    if sess.facts:
        lines.append("STICKY_NOTES (facts the user told you; treat as true unless contradicted):")
        for k, v in list(sess.facts.items())[:30]:
            lines.append(f"- {k} = {v}")

    return "\n".join(lines)

def wants_scripture(message: str) -> bool:
    m = message.lower()
    return any(w in m for w in ["verse", "scripture", "bible", "qur", "qur'an", "quran", "sutra", "[s#]"])

def detect_corpus(message: str, sess: SessionState) -> Optional[str]:
    m = message.lower()
    if "bible" in m or "christ" in m:
        return "bible"
    if "qur" in m or "islam" in m or "muslim" in m:
        return "quran"
    if "sutta" in m or "buddh" in m:
        return "sutta_pitaka"
    return sess.faith

def call_openai_chat(messages: List[Dict[str, str]], stream: bool = False) -> Generator[str, None, str]:
    """
    Returns a generator of text chunks if stream=True, else a one-shot string.
    """
    if not _HAS_OPENAI or not OPENAI_API_KEY:
        # No API → echo fake completion for dev
        text = "I’m running without OpenAI credentials right now."
        def _gen():
            yield text
        return _gen() if stream else (text)  # type: ignore

    client = OpenAI(api_key=OPENAI_API_KEY)
    if stream:
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=messages,
            temperature=0.6,
            stream=True,
        )
        def _gen():
            for ev in resp:
                chunk = ev.choices[0].delta.content or ""
                if chunk:
                    yield chunk
        return _gen()
    else:
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=messages,
            temperature=0.6,
        )
        return resp.choices[0].message.content or ""

# ---------- SSE helpers ----------
def sse_format(event: Optional[str], data: str) -> str:
    # event can be None for default 'message'
    if event:
        return f"event: {event}\n" + "\n".join(f"data: {line}" for line in data.splitlines()) + "\n\n"
    else:
        return "\n".join(f"data: {line}" for line in data.splitlines()) + "\n\n"

def smooth_stream(chunks: Generator[str, None, None], flush_chars: int = 80) -> Generator[str, None, None]:
    """
    Buffer partial tokens into readable phrases.
    Flush at sentence boundaries or after ~flush_chars.
    """
    buf = ""
    for piece in chunks:
        buf += piece
        # flush on sentence-ish boundary
        if re.search(r"[\.!\?]\s+$", buf) or len(buf) >= flush_chars:
            yield buf
            buf = ""
    if buf:
        yield buf

# ---------- FastAPI ----------
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten later if needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/diag_rag")
def diag_rag():
    return {"ok": True, "corpora": diag_rag_counts()}

@dataclass
class ChatPayload:
    message: str

def _build_messages(sess: SessionState, user_msg: str) -> List[Dict[str, str]]:
    sys = {"role": "system", "content": build_system_prompt(sess)}
    msgs: List[Dict[str, str]] = [sys]
    # keep last few turns to keep style continuity without blowing context
    for turn in sess.history[-8:]:
        msgs.append(turn)
    msgs.append({"role": "user", "content": user_msg})
    return msgs

def _scripture_flow(message: str, sess: SessionState) -> Tuple[str, List[Dict[str, Any]], Optional[str]]:
    corpus = detect_corpus(message, sess) or "bible"
    sources, best = search_scripture(message, corpus, k=5)
    if not sources or not best:
        # fall back to LLM to phrase encouragement
        messages = _build_messages(sess, f"User asked for a short verse about: {message}. You couldn't find a verse. Offer a one-line encouragement.")
        text = call_openai_chat(messages, stream=False)  # type: ignore
        return (text if isinstance(text, str) else "Keep going."), [], corpus
    # Try to extract a single-sentence "verse-ish" line from best
    # Naively: take first sentence up to ~180 chars.
    verse = re.split(r"(?<=[\.!\?])\s+", best.strip())[0]
    # Ensure it’s not huge
    if len(verse) > 220:
        verse = verse[:200].rsplit(" ", 1)[0] + "..."
    # Add [S1]
    reply = f"{verse} [S1]"
    return reply, sources, corpus

def _append_history(sess: SessionState, role: str, content: str):
    sess.history.append({"role": role, "content": content})
    # hard cap
    if len(sess.history) > 24:
        sess.history = sess.history[-24:]

def _json_response(
    response_text: str,
    sources: List[Dict[str, Any]],
    selected_corpus: Optional[str],
    remembered: Optional[Dict[str, str]],
) -> JSONResponse:
    return JSONResponse({
        "response": response_text,
        "sources": sources,
        "selected_corpus": selected_corpus,
        "remembered": remembered
    })

@app.post("/chat")
async def chat(request: Request):
    body = await request.json()
    message = (body.get("message") or "").strip()
    resp = Response()
    sid, sess = get_or_create_sid(request, resp)

    remembered = parse_and_remember(message, sess)
    faith_now = try_set_faith(message, sess)

    # direct memory Q&A (fast path)
    mem_ans = lookup_memory_answer(message, sess)
    if mem_ans:
        _append_history(sess, "user", message)
        assistant = mem_ans
        _append_history(sess, "assistant", assistant)
        jr = _json_response(assistant, [], None, remembered)
        # ensure cookie is sent
        for k, v in resp.headers.items():
            jr.headers[k] = v
        return jr

    # scripture?
    if wants_scripture(message):
        reply, sources, corpus = _scripture_flow(message, sess)
        _append_history(sess, "user", message)
        _append_history(sess, "assistant", reply)
        jr = _json_response(reply, sources, corpus, remembered)
        for k, v in resp.headers.items():
            jr.headers[k] = v
        return jr

    # normal LLM turn
    messages = _build_messages(sess, message)
    out = call_openai_chat(messages, stream=False)
    reply = out if isinstance(out, str) else "".join(list(out))
    _append_history(sess, "user", message)
    _append_history(sess, "assistant", reply)
    jr = _json_response(reply, [], sess.faith, remembered)
    for k, v in resp.headers.items():
        jr.headers[k] = v
    return jr

@app.post("/chat_sse")
async def chat_sse(request: Request):
    body = await request.json()
    message = (body.get("message") or "").strip()

    base = Response()
    sid, sess = get_or_create_sid(request, base)
    remembered = parse_and_remember(message, sess)
    try_set_faith(message, sess)

    def gen() -> Generator[str, None, None]:
        # First line to establish stream
        yield ": connected\n\n"

        # Memory fast-path?
        mem_ans = lookup_memory_answer(message, sess)
        if mem_ans:
            _append_history(sess, "user", message)
            _append_history(sess, "assistant", mem_ans)
            for chunk in re.findall(r'.{1,220}(?:\s+|$)', mem_ans):
                yield sse_format(None, chunk.strip())
            # optional meta block
            meta = json.dumps({"sources": [], "selected_corpus": sess.faith, "remembered": remembered})
            yield sse_format("sources", meta)
            return

        # Scripture?
        if wants_scripture(message):
            reply, sources, corpus = _scripture_flow(message, sess)
            _append_history(sess, "user", message)
            _append_history(sess, "assistant", reply)
            for chunk in re.findall(r'.{1,220}(?:\s+|$)', reply):
                yield sse_format(None, chunk.strip())
            meta = json.dumps({"sources": sources, "selected_corpus": corpus, "remembered": remembered})
            yield sse_format("sources", meta)
            return

        # Normal streamed LLM
        messages = _build_messages(sess, message)
        stream_chunks = call_openai_chat(messages, stream=True)  # type: ignore
        for chunk in smooth_stream(stream_chunks):  # type: ignore
            yield sse_format(None, chunk)
        _append_history(sess, "user", message)
        # We didn't buffer the whole reply; fine, history continuity is still ok for next turn.

        # Optional trailing meta
        meta = json.dumps({"sources": [], "selected_corpus": sess.faith, "remembered": remembered})
        yield sse_format("sources", meta)

    headers = dict(base.headers)
    return StreamingResponse(gen(), media_type="text/event-stream", headers=headers)

# Friendly root
@app.get("/")
def root():
    return {"ok": True, "msg": "LLM API Demo backend is running"}
