import os, json, re, uuid, traceback
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse, Response
from openai import OpenAI
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import load_npz
import joblib

# ========= OpenAI client =========
def get_client():
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY not set")
    return OpenAI(api_key=key)

# ========= CORS =========
origins = [
    "https://dturnover.github.io",  # GitHub Pages demo
    # "https://<your-wp-domain>",   # add when ready
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

When sources are provided, ground your answer ONLY in those sources, citing inline as [S1], [S2], etc. If insufficient, say so and ask for a more specific passage.
Stay in character throughout.
""".strip()

# ========= In-memory session (per instance) =========
SESSIONS: Dict[str, Dict[str, Any]] = {}  # sid -> {"religion": "bible"/"quran"/"sutta_pitaka"}

def get_or_create_sid(request: Request) -> str:
    sid = request.cookies.get("sid")
    if not sid or sid not in SESSIONS:
        sid = uuid.uuid4().hex
        SESSIONS[sid] = {"religion": None}
    return sid

RELIGION_MAP = {
    "bible": "bible", "christian": "bible", "christianity": "bible", "jesus": "bible",
    "quran": "quran", "koran": "quran", "muslim": "quran", "islam": "quran", "allah": "quran",
    "buddhist": "sutta_pitaka", "buddhism": "sutta_pitaka", "sutta": "sutta_pitaka",
    "pitaka": "sutta_pitaka", "pali": "sutta_pitaka", "dhamma": "sutta_pitaka",
}

def detect_religion(text: str) -> str | None:
    t = text.lower()
    # Simple heuristic; you can swap with a tiny LLM classifier later
    for k, corpus in RELIGION_MAP.items():
        if re.search(rf"\b{k}\b", t):
            return corpus
    # Patterns like "I am X", "I'm X"
    m = re.search(r"\b(i[' ]?m|i am)\s+(christian|muslim|buddhist|islam|buddhism|christianity)\b", t)
    if m:
        return RELIGION_MAP.get(m.group(2), None)
    return None

# ========= RAG load (multi-corpus) =========
BASE_DIR = Path(__file__).resolve().parent
RAG_DIR = BASE_DIR / "rag_data"

# corpora[name] = {
#   "docs": List[dict], "emb": np.ndarray (optional/not used here),
#   "tfidf": TfidfVectorizer, "tfidf_mat": csr_matrix
# }
CORPORA: Dict[str, Dict[str, Any]] = {}

def load_all_corpora():
    CORPORA.clear()
    if not RAG_DIR.exists():
        print("RAG dir missing; skip.")
        return
    for sub in RAG_DIR.iterdir():
        if not sub.is_dir():
            continue
        name = sub.name
        try:
            docs_path = sub / "docs.json"
            tfidf_vec_path = sub / "tfidf_vectorizer.joblib"
            tfidf_mat_path = sub / "tfidf_matrix.npz"
            if not (docs_path.exists() and tfidf_vec_path.exists() and tfidf_mat_path.exists()):
                continue
            with open(docs_path, "r", encoding="utf-8") as f:
                docs = json.load(f)
            tfidf: TfidfVectorizer = joblib.load(tfidf_vec_path)
            tfidf_mat = load_npz(tfidf_mat_path)
            CORPORA[name] = {"docs": docs, "tfidf": tfidf, "tfidf_mat": tfidf_mat}
            print(f"RAG loaded: {name} -> {len(docs)} chunks, TF-IDF {tfidf_mat.shape}")
        except Exception as e:
            print(f"RAG load error for {name}:", e)

load_all_corpora()

def retrieve_tfidf(query: str, corpus: str, k: int = 5) -> List[Dict[str, Any]]:
    c = CORPORA.get(corpus)
    if not c:
        return []
    tfidf: TfidfVectorizer = c["tfidf"]
    tfidf_mat = c["tfidf_mat"]
    docs = c["docs"]
    qv = tfidf.transform([query])                # (1, V)
    sims = (tfidf_mat @ qv.T).toarray().ravel()  # proxy similarity
    top_idx = sims.argsort()[::-1][:k]
    hits = []
    for i in top_idx:
        hits.append({
            "score": float(sims[i]),
            "id": docs[i]["id"],
            "page": docs[i].get("page"),
            "text": docs[i]["text"],
            "corpus": corpus,
        })
    return hits

def build_sources_block(hits: List[Dict[str, Any]]) -> str:
    lines = []
    for i, h in enumerate(hits, start=1):
        label = f"[S{i}]"
        page = f"p{h['page']}" if h.get("page") else ""
        lines.append(f"{label} ({page} {h['id']}) {h['text']}")
    return "\n\n".join(lines) if lines else "No sources found."

# ========= Health / diag =========
@app.get("/")
def root():
    return {"ok": True, "message": "API is running", "endpoints": ["/chat (stream)", "/chat_json", "/diag", "/diag_rag", "/chat_rag_json", "/chat_sse"]}

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
    corpora = {k: (len(v.get("docs") or [])) for k, v in CORPORA.items()}
    return {"ok": True, "corpora": corpora}

# ========= Non-stream JSON chat ====
