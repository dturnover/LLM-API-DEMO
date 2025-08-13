import os, json, traceback
from pathlib import Path

import numpy as np
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from openai import OpenAI
from sklearn.feature_extraction.text import TfidfVectorizer

# ========= OpenAI client =========
def get_client():
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY not set")
    return OpenAI(api_key=key)

# ========= CORS =========
origins = [
    "https://dturnover.github.io",         # GitHub Pages demo
    # "https://<his-domain.com>",          # add his WP domain when ready
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

Examples:
User: “How should one face adversity?”
Assistant: “With unflinching resolve, tempered by prudence. We are not made to yield at the first squall, but to tack and see the voyage through.”

User: “Tell a quick joke.”
Assistant: “I have often found that the shortest speeches require the longest preparation—mercifully for you, this one is brief.”

Stay in character throughout.
"""

# ========= RAG (precomputed) =========
BASE_DIR = Path(__file__).resolve().parent
RAG_DIR = BASE_DIR / "rag_data"
RAG_EMB = None
RAG_DOCS = None
TFIDF = None
TFIDF_MAT = None

def load_rag():
    """Load precomputed embeddings + docs and build TF-IDF matrix for retrieval."""
    global RAG_EMB, RAG_DOCS, TFIDF, TFIDF_MAT
    try:
        emb_path = RAG_DIR / "embeddings.npy"
        docs_path = RAG_DIR / "docs.json"
        if not emb_path.exists() or not docs_path.exists():
            print("RAG: embeddings/docs not found; skipping.")
            return
        RAG_EMB = np.load(emb_path)
        with open(docs_path, "r", encoding="utf-8") as f:
            RAG_DOCS = json.load(f)
        TFIDF = TfidfVectorizer(stop_words="english", max_features=20000)
        TFIDF_MAT = TFIDF.fit_transform([d["text"] for d in RAG_DOCS])
        print(f"RAG loaded: {len(RAG_DOCS)} chunks; TF-IDF shape {TFIDF_MAT.shape}")
    except Exception as e:
        print("RAG load error:", e)

load_rag()

def retrieve_tfidf(query: str, k: int = 5):
    """Lightweight lexical retrieval (good enough to demo)."""
    if TFIDF is None or TFIDF_MAT is None or not RAG_DOCS:
        return []
    qv = TFIDF.transform([query])                 # (1, V)
    sims = (TFIDF_MAT @ qv.T).toarray().ravel()   # dense similarity proxy
    top_idx = sims.argsort()[::-1][:k]
    hits = []
    for i in top_idx:
        hits.append({
            "score": float(sims[i]),
            "id": RAG_DOCS[i]["id"],
            "page": RAG_DOCS[i].get("page"),
            "text": RAG_DOCS[i]["text"],
        })
    return hits

# ========= Health / diag =========
@app.get("/")
def root():
    return {"ok": True, "message": "API is running", "endpoints": ["/chat (stream)", "/chat_json (non-stream)", "/diag", "/diag_rag", "/chat_rag_json"]}

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
    ok = RAG_DOCS is not None and TFIDF_MAT is not None
    n = len(RAG_DOCS) if RAG_DOCS else 0
    return {"ok": bool(ok), "chunks": n}

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
            model="gpt-4o-mini",
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
                model="gpt-4o-mini",
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
            yield ""  # clean close

        return StreamingResponse(gen(), media_type="text/plain")
    except Exception as e:
        traceback.print_exc()
        return JSONResponse({"error": str(e)}, status_code=500)

# ========= RAG chat (non-streaming) =========
@app.post("/chat_rag_json")
async def chat_rag_json(request: Request):
    """
    Retrieve top passages from rag_data and answer grounded in them, with inline citations [S1].. .
    """
    try:
        body = await request.json()
        user_message = (body.get("message") or "").strip()
        if not user_message:
            return JSONResponse({"error": "message is required"}, status_code=400)

        # 1) retrieve
        hits = retrieve_tfidf(user_message, k=5)

        # 2) build context with labels [S1], [S2]...
        context_lines = []
        for i, h in enumerate(hits, start=1):
            label = f"[S{i}]"
            page = f"p{h['page']}" if h.get("page") else ""
            context_lines.append(f"{label} ({page} {h['id']}) {h['text']}")
        context_block = "\n\n".join(context_lines) if context_lines else "No sources found."

        # 3) ask the model, grounded
        grounded_prompt = f"""Use ONLY the sources below to answer. If they are insufficient,
say so and ask for a more specific passage. Cite sources inline as [S1], [S2], etc.

SOURCES:
{context_block}

QUESTION:
{user_message}
"""

        client = get_client()
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.3,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT + " Always ground answers in the provided sources when available."},
                {"role": "user", "content": grounded_prompt},
            ],
        )
        answer = resp.choices[0].message.content
        return {"response": answer, "sources": hits}
    except Exception as e:
        traceback.print_exc()
        return JSONResponse({"error": str(e)}, status_code=500)
