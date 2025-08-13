# prep_rag.py
import os, json
import numpy as np
from pypdf import PdfReader
import tiktoken
from sentence_transformers import SentenceTransformer

# ---- config ----
PDF_PATH = "pdfs/The-Holy-Bible-King-James-Version.pdf"   # relative path (adjust if needed)
OUT_DIR = "rag_data"
CHUNK_TOKENS = 500
OVERLAP_TOKENS = 100
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

enc = tiktoken.get_encoding("cl100k_base")  # close-enough tokenizer for chunking

def chunk_text(text: str, chunk_tokens=CHUNK_TOKENS, overlap=OVERLAP_TOKENS):
    """Token-slide window: 500 tokens with 100-token overlap."""
    toks = enc.encode(text or "")
    chunks = []
    i = 0
    n = len(toks)
    while i < n:
        j = min(i + chunk_tokens, n)
        chunk = enc.decode(toks[i:j])
        chunks.append(chunk)
        if j == n:
            break
        i = max(0, j - overlap)
    return chunks

def read_pdf(path: str):
    reader = PdfReader(path)
    pages = []
    for i, p in enumerate(reader.pages):
        txt = p.extract_text() or ""
        if txt.strip():
            pages.append({"page": i + 1, "text": txt.strip()})
    return pages

def build():
    # 0) ensure files
    if not os.path.exists(PDF_PATH):
        raise FileNotFoundError(f"PDF not found at {PDF_PATH!r}. Put it there or change PDF_PATH.")
    os.makedirs(OUT_DIR, exist_ok=True)

    # 1) read + chunk
    print("Reading PDF…")
    pages = read_pdf(PDF_PATH)

    print("Chunking…")
    docs = []
    for p in pages:
        for idx, ch in enumerate(chunk_text(p["text"])):
            docs.append({
                "id": f"p{p['page']}_c{idx}",
                "page": p["page"],
                "text": ch
            })

    if not docs:
        raise RuntimeError("No chunks produced; is the PDF text extraction empty?")

    # 2) embed (local, free)
    print(f"Embedding {len(docs)} chunks with {MODEL_NAME}…")
    model = SentenceTransformer(MODEL_NAME)
    embs = model.encode([d["text"] for d in docs], normalize_embeddings=True)

    # 3) save
    np.save(os.path.join(OUT_DIR, "embeddings.npy"), embs.astype(np.float32))
    with open(os.path.join(OUT_DIR, "docs.json"), "w", encoding="utf-8") as f:
        json.dump(docs, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(docs)} chunks to {OUT_DIR}/ (embeddings.npy + docs.json)")

if __name__ == "__main__":
    build()
