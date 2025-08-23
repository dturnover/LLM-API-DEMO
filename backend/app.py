# app.py — FastAPI + cookie sessions + memory + tiny RAG + OpenAI + SSE
import os, re, json, uuid
from typing import Dict, List, Optional, Any, Iterable
from pathlib import Path
from dataclasses import dataclass, field
from collections import defaultdict

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse, PlainTextResponse
from pydantic import BaseModel

# ---------- OpenAI ----------
from openai import OpenAI
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL   = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
client = OpenAI() if OPENAI_API_KEY else None

# ---------- FastAPI + CORS ----------
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=".*",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

@app.middleware("http")
async def echo_origin(request: Request, call_next):
    resp: Response = await call_next(request)
    origin = request.headers.get("origin")
    if origin:
        resp.headers["Access-Control-Allow-Origin"] = origin
        resp.headers["Vary"] = (resp.headers.get("Vary", "") + ", Origin").strip(", ")
        resp.headers["Access-Control-Allow-Credentials"] = "true"
        resp.headers["Access-Control-Expose-Headers"] = "*"
        resp.headers["Access-Control-Allow-Headers"] = "*"
        resp.headers["Access-Control-Allow-Methods"] = "GET,POST,OPTIONS"
    return resp

@app.options("/{path:path}")
def preflight(path: str):  # CORS preflight
    return Response(status_code=204)

# ---------- Sessions & memory ----------
@dataclass
class Session:
    faith: Optional[str] = None            # "bible" | "quran" | "sutta_pitaka"
    facts: Dict[str, str] = field(default_factory=dict)  # normalized key -> value
SESS: Dict[str, Session] = {}

_WORDS = re.compile(r"[^a-z0-9 ]+")
def _norm(s: str) -> str:
    s = (s or "").lower().strip()
    s = " " + _WORDS.sub(" ", s) + " "
    for w in (" the ", " a ", " an ", " my ", " your "):
        s = s.replace(w, " ")
    return " ".join(s.split())

def get_or_create_sid(request: Request, response: Response) -> str:
    sid = request.cookies.get("sid")
    if not sid:
        sid = uuid.uuid4().hex
        origin = (request.headers.get("origin") or "").strip().lower()
        host = f"{request.url.scheme}://{request.url.netloc}".lower()
        cross_site = bool(origin and origin != host)
        response.set_cookie(
            "sid", sid, path="/", httponly=True,
            secure=True if cross_site else (request.url.scheme == "https"),
            samesite="none" if cross_site else "lax"
        )
    if sid not in SESS:
        SESS[sid] = Session()
    return sid

def parse_remember(msg: str) -> Optional[Dict[str, str]]:
    if not re.search(r"\bremember\b", msg, re.I):
        return None
    m = re.search(r"remember(?:\s+that)?\s+(?P<key>.+?)\s+(?:is|are|=)\s+(?P<val>.+)", msg, re.I)
    if m:
        return {"key": _norm(m.group("key")), "value": m.group("val").strip().rstrip(".")}
    body = re.sub(r"^\s*remember(?::|\s+that)?\s*", "", msg, flags=re.I).strip()
    words = body.split()
    key = _norm(" ".join(words[:4])) or "note"
    val = " ".join(words[4:]) or body
    return {"key": key, "value": val}

def memory_put(sess: Session, kv: Dict[str, str]):
    sess.facts[kv["key"]] = kv["value"]

def memory_get(sess: Session, msg: str) -> Optional[str]:
    t = _norm(msg)
    for k, v in sess.facts.items():
        if k in t:
            return v
        kt, tt = set(k.split()), set(t.split())
        if kt and (len(kt & tt) / len(kt)) >= 0.6:
            return v
    return None

FAITH_MAP = {"christian": "bible", "muslim": "quran", "buddhist": "sutta_pitaka"}
def maybe_set_faith(msg: str, sess: Session):
    m = re.search(r"\b(i am|i'm)\s+(christian|muslim|buddhist)\b", msg.lower())
    if m:
        sess.faith = FAITH_MAP[m.group(2)]

# ---------- Tiny RAG ----------
@dataclass
class Doc:
    id: str; page: int; text: str; corpus: str

CORPUS: Dict[str, List[Doc]] = defaultdict(list)
COUNTS: Dict[str, int] = defaultdict(int)

def _ingest(r: Dict[str, Any]):
    t = (r.get("text") or "").strip()
    if not t: return
    c = (r.get("corpus") or "misc").lower()
    d = Doc(
        id=str(r.get("id") or f"{c}_{r.get('page', 0)}"),
        page=int(r.get("page") or 0),
        text=t,
        corpus=c,
    )
    CORPUS[c].append(d)

def load_rag():
    root = Path(__file__).parent / "rag_data"
    if not root.exists(): return
    for p in list(root.glob("*.jsonl")) + list(root.glob("*.json")):
        try:
            if p.suffix == ".jsonl":
                for ln in p.read_text(encoding="utf-8").splitlines():
                    if not ln.strip(): continue
                    _ingest(json.loads(ln))
            else:
                data = json.loads(p.read_text(encoding="utf-8"))
                for r in (data if isinstance(data, list) else data.get("items", [])):
                    _ingest(r)
        except:
            pass
    for sub in root.iterdir():
        if not sub.is_dir(): continue
        dj = sub / "docs.json"
        if dj.exists():
            try:
                data = json.loads(dj.read_text(encoding="utf-8"))
                for r in data:
                    r.setdefault("corpus", sub.name)
                    _ingest(r)
            except:
                pass
    for k, arr in CORPUS.items():
        COUNTS[k] = len(arr)
load_rag()

def rag_query(q: str, corpus: str, k: int = 3) -> List[Doc]:
    docs = CORPUS.get(corpus, [])
    if not docs: return []
    q = re.sub(r"\[s#\]", " ", q.lower())
    q = re.sub(r"[^a-z0-9 ]+", " ", q)
    toks = [w for w in q.split() if w not in {"give","share","short","about","verse","a","the","of","and"}] or q.split()
    def score(text: str) -> int:
        t = text.lower()
        return sum(t.count(tok) for tok in toks if tok)
    ranked = sorted(((score(d.text), d) for d in docs), key=lambda x: x[0], reverse=True)
    return [d for s, d in ranked[:k] if s > 0] or (docs[:1] if docs else [])

def sources_payload(docs: List[Doc]) -> List[Dict[str, Any]]:
    out = []
    for d in docs[:5]:
        snippet = d.text if len(d.text) <= 900 else d.text[:900] + "..."
        out.append({"id": d.id, "page": d.page, "text": snippet, "corpus": d.corpus, "score": 0.0})
    return out

def detect_corpus(msg: str, sess: Session) -> Optional[str]:
    m = msg.lower()
    if "qur" in m or "koran" in m or "quran" in m: return "quran"
    if "bible" in m or "psalm" in m or "jesus" in m or "[s#" in m: return "bible"
    if "sutta" in m or "pitaka" in m or "buddha" in m: return "sutta_pitaka"
    return sess.faith

# ---------- Proactive scripture logic ----------
EMO = {
    "scared","afraid","nervous","anxious","anxiety","worry","worried","stress","stressed",
    "tired","hurt","injured","pain","sad","grief","loss","breakup","fear","doubt","weak",
    "overwhelmed","panic","angry","frustrated","discouraged","hopeless","lonely","shame","guilt"
}
SPIRIT = {
    "courage","perseverance","patience","hope","faith","forgiveness","gratitude",
    "strength","endurance","redemption","mercy","peace","calm","trust"
}
ROUTINE = {"pre-fight","before fight","before sparring","sparring","fight","weigh-in","routine","warmup","warm-up","breathe","breathing"}

def should_use_scripture(msg: str, sess: Session) -> Optional[str]:
    if not sess.faith: return None
    m = msg.lower()
    if detect_corpus(msg, sess): return detect_corpus(msg, sess)
    if any(w in m for w in EMO) or any(w in m for w in SPIRIT) or any(w in m for w in ROUTINE):
        return sess.faith
    return None

def pick_short_verse_text(docs: List[Doc]) -> str:
    if not docs: return "No passage found."
    first = docs[0].text.strip()
    line = first.split("\n")[0].strip()
    if len(line) > 240: line = line[:240].rsplit(" ", 1)[0] + "…"
    return line

def one_line_reflection(user_msg: str, verse_text: str) -> str:
    if not client:
        return "Breathe. You’re not alone; choose your pace and fight with clarity."
    r = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role":"system","content":"In ONE sentence (<=24 words), apply the verse to the user's situation with a calm mentor voice. No quotes, no citations."},
            {"role":"user","content": f"User: {user_msg}\nVerse: {verse_text}"},
        ],
        temperature=0.6,
    )
    return (r.choices[0].message.content or "").strip() or "Hold steady—courage grows with each faithful breath."

def scripture_reply(user_msg: str, corpus: str) -> Dict[str, Any]:
    docs = rag_query(user_msg, corpus, k=3)
    srcs = sources_payload(docs)
    verse = pick_short_verse_text(docs)
    refl  = one_line_reflection(user_msg, verse)
    return {"text": f"{verse} [S1]\n➝ {refl}", "sources": srcs, "corpus": corpus}

# ---------- Prompts ----------
def system_prompt(sess: Session) -> str:
    notes = "\n".join(f"- {k}: {v}" for k, v in list(sess.facts.items())[:30]) or "- (none)"
    faith = sess.faith or "(not set)"
    return (
        "You are Fight Chaplain: a calm, modern, practical pastoral coach for combat athletes. "
        "Be brief. Encourage safety and agency. If scripture not provided, stay secular.\n"
        f"Faith: {faith}\nKnown facts:\n{notes}"
    )

# ---------- Models ----------
class ChatIn(BaseModel):
    message: str

# ---------- Health ----------
@app.get("/")
def root():
    return PlainTextResponse("ok")

@app.get("/health")
def health():
    return JSONResponse({"ok": True, "sessions": len(SESS), "corpora": COUNTS})

# ---------- /chat (JSON primary) ----------
@app.post("/chat")
async def chat(payload: ChatIn, request: Request, response: Response):
    sid = get_or_create_sid(request, response); sess = SESS[sid]
    msg = (payload.message or "").strip()
    maybe_set_faith(msg, sess)

    # write memory (ack immediately)
    remembered = parse_remember(msg)
    if remembered:
        memory_put(sess, remembered)
        return {"response": f"Noted: {remembered['key']} = {remembered['value']}.", "sources": [], "selected_corpus": None, "remembered": remembered}

    # memory fast-path read
    mem = memory_get(sess, msg)
    if mem:
        subj = next((k for k in sess.facts if k in _norm(msg)), None)
        text = f"{subj} is {mem}." if subj and not mem.startswith(("is ","are ")) else (mem if subj is None else f"{subj} {mem}.")
        return {"response": text, "sources": [], "selected_corpus": None, "remembered": None}

    # proactive scripture when faith known + emotional/spiritual intent
    auto_corpus = should_use_scripture(msg, sess)
    if auto_corpus:
        rep = scripture_reply(msg, auto_corpus)
        return {"response": rep["text"], "sources": rep["sources"], "selected_corpus": rep["corpus"], "remembered": None}

    # generic → OpenAI
    text = "How can I help further?"
    try:
        if client:
            r = client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[{"role":"system","content": system_prompt(sess)}, {"role":"user","content": msg}],
                temperature=0.6,
            )
            text = (r.choices[0].message.content or "").strip() or text
    except Exception as e:
        text = f"Sorry—model call failed: {e}"
    return {"response": text, "sources": [], "selected_corpus": None, "remembered": None}

# ---------- /chat_sse_get (GET SSE; optional) ----------
def _split_tokens(t: str) -> Iterable[str]:
    for w in re.split(r"(\s+)", t):
        if w: yield w

@app.get("/chat_sse_get")
async def chat_sse_get(request: Request, q: str):
    sid_resp = Response(); sid = get_or_create_sid(request, sid_resp); sess = SESS[sid]
    msg = (q or "").strip()
    maybe_set_faith(msg, sess)

    # allow "remember ..." via SSE too
    remembered = parse_remember(msg)
    if remembered: memory_put(sess, remembered)

    # build response text
    mem = memory_get(sess, msg)
    final_meta = None
    if mem:
        subj = next((k for k in sess.facts if k in _norm(msg)), None)
        full = f"{subj} is {mem}." if subj and not mem.startswith(("is ","are ")) else (mem if subj is None else f"{subj} {mem}.")
    else:
        auto_corpus = should_use_scripture(msg, sess)
        if auto_corpus:
            rep = scripture_reply(msg, auto_corpus)
            full = rep["text"]
            final_meta = {"sources": rep["sources"], "selected_corpus": rep["corpus"]}
        else:
            try:
                if client:
                    r = client.chat.completions.create(
                        model=OPENAI_MODEL,
                        messages=[{"role":"system","content": system_prompt(sess)}, {"role":"user","content": msg}],
                        temperature=0.6,
                    )
                    full = (r.choices[0].message.content or "").strip() or "How can I help further?"
                else:
                    full = "I’m running without OpenAI credentials."
            except Exception as e:
                full = f"Sorry—model call failed: {e}"

    async def gen():
        yield ": connected\n\n"
        sent_any = False
        for tok in _split_tokens(full):
            sent_any = True
            yield f"data: {tok}\n\n"
        if not sent_any:
            yield "data: \n\n"  # force flush through proxies
        if final_meta:
            yield f"event: sources\ndata: {json.dumps(final_meta)}\n\n"
        yield "event: done\ndata: {}\n\n"

    headers = {
        "Cache-Control": "no-store, no-cache, no-transform",
        "X-Accel-Buffering": "no",
        "Connection": "keep-alive",
    }
    headers.update(sid_resp.headers)
    return StreamingResponse(gen(), media_type="text/event-stream", headers=headers)

# ---------- dev run ----------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=int(os.getenv("PORT","8000")), reload=False)
