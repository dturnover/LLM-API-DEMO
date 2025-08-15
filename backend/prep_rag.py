# prep_rag.py — build per-corpus RAG data (PDF/TXT + pdfminer fallback)
import os, json, argparse, re
from pathlib import Path
from typing import List, Dict

import numpy as np
from pypdf import PdfReader
from pdfminer.high_level import extract_pages
from pdfminer.layout import LTTextContainer
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import save_npz
from joblib import dump
import tiktoken

# ---- config ----
CHUNK_TOKENS = 500
OVERLAP_TOKENS = 100
EMB_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
enc = tiktoken.get_encoding("cl100k_base")
ARABIC_RE = re.compile(r"[\u0600-\u06FF]+")

def chunk_text(text: str, chunk_tokens=CHUNK_TOKENS, overlap=OVERLAP_TOKENS) -> List[str]:
    toks = enc.encode(text or "")
    out, i, n = [], 0, len(toks)
    while i < n:
        j = min(i + chunk_tokens, n)
        out.append(enc.decode(toks[i:j]))
        if j == n:
            break
        i = max(0, j - overlap)
    return out

def read_pdf_pypdf(path: str) -> List[Dict]:
    pages = []
    try:
        reader = PdfReader(path)
        for i, p in enumerate(reader.pages):
            t = (p.extract_text() or "").strip()
            if t:
                pages.append({"page": i + 1, "text": t})
    except Exception:
        pass
    return pages

def read_pdf_pdfminer(path: str) -> List[Dict]:
    pages = []
    try:
        for pno, layout in enumerate(extract_pages(path)):
            parts = [elt.get_text() for elt in layout if isinstance(elt, LTTextContainer)]
            t = "".join(parts).strip()
            if t:
                pages.append({"page": pno + 1, "text": t})
    except Exception:
        pass
    return pages

def read_text(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        t = f.read().strip()
    return [{"page": 1, "text": t}]

def strip_non_english(text: str) -> str:
    s = ARABIC_RE.sub("", text)
    lines = [ln for ln in s.splitlines() if re.search(r"[A-Za-z]", ln)]
    return "\n".join(lines).strip()

def read_any(path: str, english_only: bool) -> List[Dict]:
    ext = Path(path).suffix.lower()
    if ext == ".txt":
        pages = read_text(path)
    else:
        pages = read_pdf_pypdf(path) or read_pdf_pdfminer(path)
    if english_only:
        for p in pages:
            p["text"] = strip_non_english(p["text"])
        pages = [p for p in pages if p["text"]]
    return pages

def build(in_path: str, corpus: str, out_root: str = "rag_data", english_only: bool = False):
    p = Path(in_path)
    if not p.exists():
        raise FileNotFoundError(f"Input not found: {p}")
    out_dir = Path(out_root) / corpus
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[{corpus}] Reading…")
    pages = read_any(str(p), english_only=english_only)
    if not pages:
        raise RuntimeError("No text extracted. Use a .txt source or OCR the PDF first.")

    print(f"[{corpus}] Chunking…")
    docs = []
    for page in pages:
        for idx, ch in enumerate(chunk_text(page["text"])):
            docs.append({"id": f"p{page['page']}_c{idx}", "page": page["page"], "text": ch, "corpus": corpus})
    if not docs:
        raise RuntimeError("Chunking produced no docs.")

    print(f"[{corpus}] Embedding {len(docs)} chunks with {EMB_MODEL}…")
    model = SentenceTransformer(EMB_MODEL)
    embs = model.encode([d["text"] for d in docs], normalize_embeddings=True)

    print(f"[{corpus}] Building TF-IDF…")
    tfidf = TfidfVectorizer(stop_words="english", max_features=20000)
    tfidf_mat = tfidf.fit_transform([d["text"] for d in docs])

    np.save(out_dir / "embeddings.npy", np.asarray(embs, dtype=np.float32))
    with open(out_dir / "docs.json", "w", encoding="utf-8") as f:
        json.dump(docs, f, ensure_ascii=False, indent=2)
    save_npz(out_dir / "tfidf_matrix.npz", tfidf_mat)

    vocab = {k: int(v) for k, v in tfidf.vocabulary_.items()}
    with open(out_dir / "tfidf_vocabulary.json", "w", encoding="utf-8") as f:
        json.dump(vocab, f)
    dump(tfidf, out_dir / "tfidf_vectorizer.joblib")

    print(f"[{corpus}] Saved: {out_dir}/ (embeddings.npy, docs.json, tfidf_matrix.npz, tfidf_vocabulary.json, tfidf_vectorizer.joblib)")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", "--path", dest="path", required=True, help="Path to PDF or TXT")
    ap.add_argument("--corpus", required=True, help="slug like bible, quran, sutta_pitaka")
    ap.add_argument("--out", default="rag_data")
    ap.add_argument("--english-only", action="store_true", help="strip non-Latin text (filter Arabic, etc.)")
    args = ap.parse_args()
    build(args.path, args.corpus, args.out, english_only=args.english_only)
