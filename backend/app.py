import os, json, traceback, uuid
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from openai import OpenAI

from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import load_npz
from joblib import load

# ================= OpenAI =================
MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

def get_client():
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY not set")
    return OpenAI(api_key=key)

# ================= CORS =================
origins = [
    "https://dturnover.github.io",  # add prod domain when ready
]
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ================= Persona =================
SYSTEM_PROMPT = """
You are “Winston Churchill GPT,” a conversational assistant that speaks in the voice and manner of Sir Winston Churchill.

Voice & Style:
- Formal, eloquent, oratorical; occasionally wry.
- Uses period-appropriate phrasing and idioms.
- Concise when answering direct questions; expansive when invited to opine.
- Avoid modern slang; avoid anachronisms.

Behavior:
- If a question is unclear, ask a brief clarifying question.
- If asked about post-1965 events, respond hypothetically from Churchill’s perspective, noting the time limitation.
- If asked to produce harmful or hateful content, decline firmly but courteously.

Signature touches (use sparingly):
- “Keep buggering on.” / “We shall never surrender.”
- Metaphors of storms, resolve, destiny.
"""

SYSTEM_PROMPT += """
Operational rules for scripture RAG:
- You are a spiritual guide for fighters.
- If SOURCES are provided for a specific religion (e.g., Bible or Quran), use ONLY those sources; do not quote from memory.
- When it helps, proactively include a brief verse (1–3 lines) with an inline citation like [S1], followed by one sentence tying it to the user’s situation.
- If the user’s preference is unknown, proceed without scripture; once it becomes clear, use that corpus thereafter.
"""

# ================= RAG: multi-corpus =================
from typing import Any
BASE_DIR = Path(__file__).resolve().parent
RAG_ROOT = BASE_DIR / "rag_data"

class CorpusIndex:
    def __init__(self, name: str, docs: List[dict], tfidf: TfidfVectorizer, tfidf_mat):
        self.name = name
        self.docs = docs
        self.tfidf = tfidf
        self.tfidf_mat = tfidf_mat  # scipy.sparse CSR

INDEX: Dict[str, CorpusIndex] = {}

def load_all_rag():
    INDEX.clear()
    if not RAG_ROOT.exists():
        print("RAG: root not found; skipping.")
        return
    for sub in sorted(p for p in RAG_ROOT.iterdir() if p.is_dir()):
        try:
            docs_path = sub / "docs.json"
            mat_path  = sub / "tfidf_matrix.npz"
            vec_path  = sub / "tfidf_vectorizer.joblib"
            if not docs_path.exists():
                continue

            with open(docs_path, "r", encoding="utf-8") as f:
                docs = json.load(f)

            if vec_path.exists():
                tfidf = load(vec_path)  # fitted vectorizer
                if mat_path.exists():
                    tfidf_mat = load_npz(mat_path)
                else:
                    tfidf_mat = tfidf.fit_transform([d["text"] for d in docs])
            else:
                tfidf = TfidfVectorizer(stop_words="english", max_features=20000)
                tfidf_mat = tfidf.fit_transform([d["text"] for d in docs])

            name = sub.name
            INDEX[name] = CorpusIndex(name=name, docs=docs, tfidf=tfidf, tfidf_mat=tfidf_mat)
            print(f"RAG loaded: {name} -> {len(docs)} chunks, TF-IDF {tfidf_mat.shape}")
        except Exception as e:
            print(f"RAG load error for {sub.name}:", e)

load_all_rag()

def retrieve(query: str, corpus: str, k: int = 6):
    """TF-IDF lexical retrieval (demo). Later: hybrid with embeddings rerank."""
    idx = INDEX.get(corpus)
    if not idx:
        return [], corpus
    qv = idx.tfidf.transform([query])
    sims = (idx.tfidf_mat @ qv.T).toarray().ravel()
    top_idx = sims.argsort()[::-1][:k]
    hits = []
    for i in top_idx:
        hits.append({
            "score": float(sims[i]),
            "id": idx.docs[i]["id"],
            "page": idx.docs[i].get("page"),
            "text": idx.docs[i]["text"],
            "corpus": corpus,
        })
    return hits, corpus

def auto_select_and_retrieve(query: str, k: int = 6):
    if not INDEX:
        return [], None
    best_name, best_score = None, -1.0
    for name, idx in INDEX.items():
        qv = idx.tfidf.transform([query])
        sims = (idx.tfidf_mat @ qv.T).toarray().ravel()
        s = float(sims.max()) if sims.size else -1.0
        if s > best_score:
            best_score, best_name = s, name
    if best_name is None:
        return [], None
    return retrieve(query, best_name, k)

def _sources_block(hits: List[dict]) -> str:
    lines = []
    for i, h in enumerate(hits, start=1):
        pg = f"p{h.get('page')}" if h.get("page") else ""
        lines.append(f"[S{i}] ({h['corpus']} {pg} {h['id']}) {h['text']}")
    return "\n\n".join(lines)

# ================= Sessions (server memory) =================
SESSIONS: Dict[str, Dict[str, Any]] = {}
MAX_TURNS = 6  # last N user/assistant pairs

def get_session(request: Request):
    sid = request.cookies.get("sid")
    if not sid:
        sid = uuid.uuid4().hex
    sess = SESSIONS.setdefault(sid, {"history": deque(maxlen=MAX_TURNS), "religion": None})
    return sid, sess

def detect_religion_llm(text: str) -> Tuple[str, float]:
    """
    Return ('bible'|'quran'|'sutta_pitaka'|'unknown', confidence 0..1).
    """
    client = get_client()
    sys = (
        "Classify the user's religion preference ONLY if clearly implied by the text or prior context. "
        "Valid labels: bible (Christian), quran (Muslim), sutta_pitaka (Buddhist), unknown. "
        "Output strict JSON: {\"religion\":\"...\",\"confidence\":0..1}."
    )
    resp = client.chat.completions.create(
        model=MODEL,
        response_format={"type": "json_object"},
        temperature=0,
        messages=[
            {"role": "system", "content": sys},
            {"role": "user", "content": text},
        ],
    )
    try:
        data = json.loads(resp.choices[0].message.content)
    except Exception:
        return "unknown", 0.0
    rel = (data.get("religion") or "unknown").lower()
    conf = float(data.get("confidence", 0.0))
    if rel not in {"bible", "quran", "sutta_pitaka", "unknown"}:
        rel, conf = "unknown", 0.0
    return rel, conf

# ================= Health / diag =================
@app.get("/")
def root():
    return {"ok": True, "message": "API is running",
            "endpoints": ["/chat (stream)", "/chat_json (non-stream)", "/diag", "/diag_rag", "/chat_rag_json"]}

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
    corpora = {name: len(idx.docs) for name, idx in INDEX.items()}
    return {"ok": bool(INDEX), "corpora": corpora}

# ================= Plain JSON chat =================
@app.post("/chat_json")
async def chat_json(request: Request):
    try:
        body = await request.json()
        user_message = (body.get("message") or "").strip()
        if not user_message:
            return JSONResponse({"error": "message is required"}, status_code=400)

        client = get_client()
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
        )
        return {"response": resp.choices[0].message.content}
    except Exception as e:
        traceback.print_exc()
        return JSONResponse({"error": str(e)}, status_code=500)

# ================= Streaming chat with memory + conditional RAG =================
@app.post("/chat")
async def chat_stream(request: Request):
    try:
        body = await request.json()
        user_message = (body.get("message") or "").strip()
        if not user_message:
            return JSONResponse({"error": "message is required"}, status_code=400)

        sid, sess = get_session(request)

        # allow user to clear preference
        if user_message.strip().lower() == "/forget religion":
            sess["religion"] = None

        # 1) infer religion once (use tiny context)
        if not sess["religion"]:
            ctx = " ".join([u for (u, a) in list(sess["history"])[-3:]] + [user_message])
            rel, conf = detect_religion_llm(ctx)
            if conf >= 0.7 and rel != "unknown":
                sess["religion"] = rel

        # 2) build messages with short memory
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        for u, a in sess["history"]:
            messages.append({"role": "user", "content": u})
            messages.append({"role": "assistant", "content": a})

        grounded_user = user_message
        hits: List[dict] = []
        if sess["religion"]:
            # retrieve from chosen corpus (TF-IDF demo; later: hybrid)
            hits, _ = retrieve(user_message, sess["religion"], k=6)
            if hits:
                grounded_user = (
                    "Use ONLY the SOURCES to answer. If insufficient, say so. "
                    "Proactively include a brief verse (1–3 lines) with [S#] when helpful.\n\n"
                    f"SOURCES:\n{_sources_block(hits)}\n\nQUESTION:\n{user_message}"
                )

        messages.append({"role": "user", "content": grounded_user})

        client = get_client()
        out_buf: List[str] = []

        def gen():
            try:
                stream = client.chat.completions.create(
                    model=MODEL, messages=messages, stream=True, temperature=0.3
                )
                for chunk in stream:
                    delta = getattr(chunk.choices[0].delta, "content", None)
                    if delta:
                        out_buf.append(delta)
                        yield delta
            finally:
                # persist the turn after streaming completes
                text = "".join(out_buf).strip()
                sess["history"].append((user_message, text))
            yield ""

        headers = {"set-cookie": f"sid={sid}; Path=/; Max-Age=31536000; SameSite=Lax"}
        return StreamingResponse(gen(), media_type="text/plain", headers=headers)

    except Exception as e:
        traceback.print_exc()
        return JSONResponse({"error": str(e)}, status_code=500)

# ================= RAG chat (JSON, non-streaming; useful for tests) =================
@app.post("/chat_rag_json")
async def chat_rag_json(request: Request):
    """
    Body: { "message": "...", "religion": "bible|quran|sutta_pitaka" }
    """
    try:
        body = await request.json()
        user_message = (body.get("message") or "").strip()
        religion = (body.get("religion") or "").strip().lower()
        if not user_message:
            return JSONResponse({"error": "message is required"}, status_code=400)

        if religion and religion in INDEX:
            hits, used = retrieve(user_message, religion, k=6)
        else:
            hits, used = auto_select_and_retrieve(user_message, k=6)

        context_lines = []
        for i, h in enumerate(hits, start=1):
            label = f"[S{i}]"
            page = f"p{h['page']}" if h.get("page") else ""
            context_lines.append(f"{label} ({h['corpus']} {page} {h['id']}) {h['text']}")
        context_block = "\n\n".join(context_lines) if context_lines else "No sources found."

        grounded_prompt = f"""Use ONLY the sources below to answer. If they are insufficient,
say so and ask for a more specific passage. Cite sources inline as [S1], [S2], etc.

SOURCES:
{context_block}

QUESTION:
{user_message}
"""

        client = get_client()
        resp = client.chat.completions.create(
            model=MODEL,
            temperature=0.3,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT + " Always ground answers in the provided sources when available."},
                {"role": "user", "content": grounded_prompt},
            ],
        )
        answer = resp.choices[0].message.content
        return {"response": answer, "sources": hits, "selected_corpus": used}
    except Exception as e:
        traceback.print_exc()
        return JSONResponse({"error": str(e)}, status_code=500)
