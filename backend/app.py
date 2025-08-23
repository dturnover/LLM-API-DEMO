# app.py — FastAPI + cookie sessions + memory + tiny RAG + OpenAI + JSON (default) + GET-SSE (optional)
import os, re, json, uuid
from typing import Dict, List, Optional, Any, Iterable
from pathlib import Path
from dataclasses import dataclass, field
from collections import defaultdict

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse, PlainTextResponse
from pydantic import BaseModel

# --- OpenAI (new SDK). Set OPENAI_API_KEY in Render. Optional OPENAI_MODEL env. ---
from openai import OpenAI
client = OpenAI()
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# --- FastAPI + CORS (force echo Origin so GH Pages can call us) ---
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
async def force_cors(request: Request, call_next):
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
def preflight(path: str): return Response(status_code=204, headers={"Cache-Control":"no-store"})
@app.head("/{path:path}")
def head_ok(path: str):  return Response(status_code=200, headers={"Cache-Control":"no-store"})

# --- Sessions & memory (cookie-scoped, in-memory) ---
@dataclass
class Session:
    faith: Optional[str] = None                       # 'bible' | 'quran' | 'sutta_pitaka'
    facts: Dict[str, str] = field(default_factory=dict)  # normalized key -> value

SESS: Dict[str, Session] = {}
_WORDS = re.compile(r"[^a-z0-9 ]+")
def _norm(s: str) -> str:
    s = (s or "").lower().strip()
    s = " " + _WORDS.sub(" ", s) + " "
    for w in (" the ", " a ", " an ", " my ", " your "): s = s.replace(w, " ")
    return " ".join(s.split())

def get_or_create_sid(request: Request, response: Response) -> str:
    sid = request.cookies.get("sid")
    if not sid:
        sid = uuid.uuid4().hex
        # cross-site cookie for GH Pages → SameSite=None; Secure
        origin = (request.headers.get("origin") or "").strip().lower()
        host = f"{request.url.scheme}://{request.url.netloc}".lower()
        cross_site = bool(origin and origin != host)
        response.set_cookie("sid", sid, path="/", httponly=True,
                            secure=True if cross_site else (request.url.scheme=="https"),
                            samesite="none" if cross_site else "lax")
    if sid not in SESS: SESS[sid] = Session()
    return sid

# --- Tiny RAG (reads rag_data/<corpus>/docs.json produced by prep_rag.py) ---
@dataclass
class Doc:
    id: str; page: int; text: str; corpus: str

CORPUS: Dict[str, List[Doc]] = defaultdict(list)
COUNTS: Dict[str, int] = defaultdict(int)

def load_rag():
    root = Path(__file__).parent / "rag_data"
    if not root.exists(): return
    # root files (optional)
    for p in list(root.glob("*.jsonl")) + list(root.glob("*.json")):
        try:
            if p.suffix==".jsonl":
                for ln in p.read_text(encoding="utf-8").splitlines():
                    if not ln.strip(): continue
                    r = json.loads(ln); _ingest(r)
            else:
                data = json.loads(p.read_text(encoding="utf-8"))
                for r in (data if isinstance(data, list) else data.get("items", [])): _ingest(r)
        except: pass
    # subfolders: rag_data/bible/docs.json etc.
    for sub in root.iterdir():
        if not sub.is_dir(): continue
        dj = sub / "docs.json"
        if dj.exists():
            try:
                for r in json.loads(dj.read_text(encoding="utf-8")): r.setdefault("corpus", sub.name); _ingest(r)
            except: pass
    for k, arr in CORPUS.items(): COUNTS[k] = len(arr)

def _ingest(rec: Dict[str, Any]):
    t = (rec.get("text") or "").strip()
    if not t: return
    c = (rec.get("corpus") or "misc").lower()
    d = Doc(id=str(rec.get("id") or f"{c}_{rec.get('page',0)}"),
            page=int(rec.get("page") or 0), text=t, corpus=c)
    CORPUS[c].append(d)

def rag_query(q: str, corpus: str, k: int=3) -> List[Doc]:
    docs = CORPUS.get(corpus, [])
    if not docs: return []
    q = re.sub(r"\[s#\]", " ", q.lower()); q = re.sub(r"[^a-z0-9 ]+", " ", q)
    toks = [w for w in q.split() if w not in {"give","share","short","about","verse","a","the","of","and"}] or q.split()
    def score(text): 
        t=text.lower(); return sum(t.count(tok) for tok in toks if tok)
    ranked = sorted(((score(d.text), d) for d in docs), key=lambda x: x[0], reverse=True)
    hit = [d for s,d in ranked[:k] if s>0] or docs[:1]
    return hit

def sources_payload(docs: List[Doc]) -> List[Dict[str, Any]]:
    out=[]
    for d in docs[:5]:
        snippet = d.text if len(d.text)<=900 else d.text[:900]+"..."
        out.append({"id":d.id,"page":d.page,"text":snippet,"corpus":d.corpus,"score":0.0})
    return out

load_rag()

# --- Helpers ---
FAITH_MAP = {"christian":"bible","muslim":"quran","buddhist":"sutta_pitaka"}
def maybe_set_faith(msg: str, sess: Session):
    m = re.search(r"\b(i am|i'm)\s+(christian|muslim|buddhist)\b", msg.lower())
    if m: sess.faith = FAITH_MAP[m.group(2)]

def detect_corpus(msg: str, sess: Session) -> Optional[str]:
    m = msg.lower()
    if "qur" in m or "koran" in m or "quran" in m: return "quran"
    if "bible" in m or "psalm" in m or "jesus" in m or "[s#" in m: return "bible"
    if "sutta" in m or "pitaka" in m or "buddha" in m: return "sutta_pitaka"
    return sess.faith

def parse_remember(msg: str) -> Optional[Dict[str,str]]:
    if not re.search(r"\bremember\b", msg, re.I): return None
    # "remember that X is Y" | "remember X is Y" | "remember: X is Y"
    m = re.search(r"remember(?:\s+that)?\s+(?P<key>.+?)\s+(?:is|are|=)\s+(?P<val>.+)", msg, re.I)
    if m:
        key, val = _norm(m.group("key")), m.group("val").strip().rstrip(".")
        return {"key":key,"value":val}
    # fallback: store whole clause under first 4 words
    body = re.sub(r"^\s*remember(?::|\s+that)?\s*", "", msg, flags=re.I).strip()
    words = body.split(); key=_norm(" ".join(words[:4])) or "note"; val=" ".join(words[4:]) or body
    return {"key":key,"value":val}

def memory_put(sess: Session, kv: Dict[str,str]): sess.facts[kv["key"]] = kv["value"]
def memory_get(sess: Session, msg: str) -> Optional[str]:
    t = _norm(msg); 
    for k,v in sess.facts.items():
        if k in t: return v
        kt, tt = set(k.split()), set(t.split())
        if kt and (len(kt&tt)/len(kt))>=0.6: return v
    return None

def system_prompt(sess: Session) -> str:
    notes="\n".join(f"- {k}: {v}" for k,v in list(sess.facts.items())[:30]) or "- (none)"
    return (
        "You are Fight Chaplain: a calm, modern, practical pastoral coach for combat athletes. "
        "Be brief by default. Encourage safety and agency. Scripture only when asked.\n"
        f"Known facts:\n{notes}"
    )

# --- Models ---
class ChatIn(BaseModel): message: str

# --- Health ---
@app.get("/")
def root():   return PlainTextResponse("ok", headers={"Cache-Control":"no-store"})
@app.get("/health")
def health(): return JSONResponse({"ok":True,"sessions":len(SESS),"corpora":COUNTS})
@app.get("/diag_rag")
def diag():   return {"ok":True,"corpora":COUNTS}

# --- /chat (JSON; default path) ---
@app.post("/chat")
async def chat(payload: ChatIn, request: Request, response: Response):
    sid = get_or_create_sid(request, response); sess = SESS[sid]
    msg = payload.message.strip()
    maybe_set_faith(msg, sess)

    remembered = parse_remember(msg)
    if remembered: memory_put(sess, remembered)

    # Fast path: memory Qs like "what color is X?"
    mem = memory_get(sess, msg)
    if mem:
        subj = next((k for k in sess.facts if k in _norm(msg)), None)
        text = f"{subj} is {mem}." if subj and not mem.startswith(("is ","are ")) else (mem if subj is None else f"{subj} {mem}.")
        return {"response": text, "sources": [], "selected_corpus": None, "remembered": None}

    # Scripture path via tiny RAG if asked or implied
    sel = detect_corpus(msg, sess)
    if sel:
        docs = rag_query(msg, sel, k=3)
        srcs = sources_payload(docs)
        quote = (docs[0].text.strip().split("\n")[0] if docs else "No passage found.")
        if len(quote)>240: quote = quote[:240].rsplit(" ",1)[0]+"…"
        return {"response": f"{quote} [S1]", "sources": srcs, "selected_corpus": sel, "remembered": remembered}

    # Generic path -> OpenAI
    try:
        messages = [
            {"role":"system","content": system_prompt(sess)},
            {"role":"user","content": msg},
        ]
        resp = client.chat.completions.create(model=OPENAI_MODEL, messages=messages, temperature=0.6)
        text = (resp.choices[0].message.content or "").strip() or "How can I help further?"
    except Exception as e:
        text = f"Sorry—model call failed: {e}"
    return {"response": text, "sources": [], "selected_corpus": None, "remembered": remembered}

# --- /chat_sse_get (GET SSE; infra-dependent) ---
def _split_tokens(t: str) -> Iterable[str]:
    for w in re.split(r"(\s+)", t):
        if w: yield w

@app.get("/chat_sse_get")
async def chat_sse_get(request: Request, q: str):
    sid_resp = Response(); sid = get_or_create_sid(request, sid_resp); sess = SESS[sid]
    msg = (q or "").strip()

    # Memory
    mem = memory_get(sess, msg)
    if mem:
        subj = next((k for k in sess.facts if k in _norm(msg)), None)
        full = f"{subj} is {mem}." if subj and not mem.startswith(("is ","are ")) else (mem if subj is None else f"{subj} {mem}.")
        async def gen():
            yield ": connected\n\n"
            for tok in _split_tokens(full): yield f"data: {tok}\n\n"
            yield "event: done\ndata: {}\n\n"
        return StreamingResponse(gen(), media_type="text/event-stream")

    # Scripture
    sel = detect_corpus(msg, sess)
    if sel:
        docs = rag_query(msg, sel, k=3)
        srcs = sources_payload(docs)
        quote = (docs[0].text.strip().split("\n")[0] if docs else "No passage found.")
        if len(quote)>240: quote = quote[:240].rsplit(" ",1)[0]+"…"
        full = f"{quote} [S1]"
        async def gen():
            yield ": connected\n\n"
            for tok in _split_tokens(full): yield f"data: {tok}\n\n"
            yield f"event: sources\ndata: {json.dumps({'sources':srcs,'selected_corpus':sel})}\n\n"
            yield "event: done\ndata: {}\n\n"
        return StreamingResponse(gen(), media_type="text/event-stream")

    # Generic via OpenAI (one-shot then tokenized for simplicity)
    try:
        messages = [
            {"role":"system","content": system_prompt(sess)},
            {"role":"user","content": msg},
        ]
        r = client.chat.completions.create(model=OPENAI_MODEL, messages=messages, temperature=0.6)
        full = (r.choices[0].message.content or "").strip() or "How can I help further?"
    except Exception as e:
        full = f"Sorry—model call failed: {e}"

    async def gen():
        yield ": connected\n\n"
        for tok in _split_tokens(full): yield f"data: {tok}\n\n"
        yield "event: done\ndata: {}\n\n"
    return StreamingResponse(gen(), media_type="text/event-stream")

# --- dev run ---
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")), reload=False)
