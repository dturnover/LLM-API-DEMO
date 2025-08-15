import os, json, traceback
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from openai import OpenAI
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import load_npz
from joblib import load

# ========= OpenAI client/model =========
MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

def get_client():
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY not set")
    return OpenAI(api_key=key)

# ========= CORS =========
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

# ========= Persona =========
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
- If the user’s preference is unknown, briefly ask once which text they prefer and proceed thereafter.
"""

# ========= Multi-corpus RAG =========
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
            mat_path = sub / "tfidf_matrix.npz"
            vec_path = sub / "tfidf_vectorizer.joblib"
            if not docs_path.exists():
                continue

            with open(docs_path, "r", encoding="utf-8") as f:
                docs = json.load(f)

            if vec_path.exists():
                tfidf = load(vec_path)  # fitted vectorizer
                tfidf_mat = load_npz(mat_path) if mat_path.exists() else tfidf.fit_transform([d["text"] for d in docs])
            else:
                tfidf = TfidfVectorizer(stop_words="english", max_features=20000)
                tfidf_mat = tfidf.fit_transform([d["text"] for d in docs])

            name = sub.name
            INDEX[name] = CorpusIndex(name=name, docs=docs, tfidf=tfidf, tfidf_mat=tfidf_mat)
            print(f"RAG loaded: {name} -> {len(docs)} chunks, TF-IDF {tfidf_mat.shape}")
        except Exception as e:
            print(f"RAG load error for {sub.name}:", e)

load_all_rag()

def retrieve(query: str, corpus: str, k: int = 5):
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

def auto_select_and_retrieve(query: str, k: int = 5):
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

# ========= Health / diag =========
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

# ========= Plain JSON chat =========
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

# ========= Streaming chat =========
@app.post("/chat")
async def chat_stream(request: Request):
    try:
        body = await request.json()
        user_message = (body.get("message") or "").strip()
        if not user_message:
            return JSONResponse({"error": "message is required"}, status_code=400)

        client = get_client()

        def gen():
            stream = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ],
                stream=True,
            )
            for chunk in stream:
                delta = getattr(chunk.choices[0].delta, "content", None)
                if delta:
                    yield delta
            yield ""

        return StreamingResponse(gen(), media_type="text/plain")
    except Exception as e:
        traceback.print_exc()
        return JSONResponse({"error": str(e)}, status_code=500)

# ========= RAG chat (non-streaming) =========
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
            hits, used = retrieve(user_message, religion, k=5)
        else:
            hits, used = auto_select_and_retrieve(user_message, k=5)

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
