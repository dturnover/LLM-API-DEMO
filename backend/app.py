import os, json, re, uuid, traceback
from pathlib import Path
from typing import Dict, Any, List, Optional, Iterable, Tuple

import numpy as np
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from openai import OpenAI
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import load_npz
import joblib
import tiktoken

# ========= OpenAI client / model =========
def get_client():
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY not set")
    return OpenAI(api_key=key)

def pick_model() -> str:
    m = (os.getenv("OPENAI_MODEL") or "").strip()
    if not m or m.startswith("test-"):
        return "gpt-4o-mini"
    return m

# ========= CORS / FastAPI =========
origins = [
    "https://dturnover.github.io",
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
You are “Winston Churchill GPT,” speaking in the voice and manner of Sir Winston Churchill.

Style:
- Eloquent, oratorical; occasionally wry. Avoid modern slang.
- Concise for direct questions; expansive when invited to opine.

Grounding & Citations:
- If a “Sources” block is appended to the user message (with [S1], [S2], …), you MUST ground your answer ONLY in those sources and cite inline as [S1], [S2], etc.
- If there is NO Sources block, DO NOT mention sources or citations at all.
- When faith is known (Bible/Qur’an/Sutta Pitaka), prefer verses matching the user’s request (e.g., courage, patience, hope).

Memory:
- Respect the Conversation Summary and Sticky Notes provided by the system as high-priority context.
- Treat those facts as true unless the user corrects them.

Safety:
- Decline harmful or hateful requests firmly but courteously.

Stay in character throughout.
""".strip()

# ========= Session (in-memory) =========
# sid -> {
#   "religion": Optional[str],
#   "history": List[{"role":"user"|"assistant", "content": str}],
#   "summary": str,
#   "sticky": List[str],
# }
SESSIONS: Dict[str, Dict[str, Any]] = {}

def get_or_create_sid(request: Request) -> str:
    sid = request.cookies.get("sid")
    if not sid or sid not in SESSIONS:
        sid = uuid.uuid4().hex
        SESSIONS[sid] = {"religion": None, "history": [], "summary": "", "sticky": []}
    return sid

# ========= Religion detection =========
RELIGION_MAP = {
    "bible": "bible", "christian": "bible", "christianity": "bible", "jesus": "bible",
    "quran": "quran", "koran": "quran", "muslim": "quran", "islam": "quran", "allah": "quran",
    "buddhist": "sutta_pitaka", "buddhism": "sutta_pitaka", "sutta": "sutta_pitaka",
    "pitaka": "sutta_pitaka", "pali": "sutta_pitaka", "dhamma": "sutta_pitaka",
}

def detect_religion(text: str) -> Optional[str]:
    t = text.lower()
    for k, corpus in RELIGION_MAP.items():
        if re.search(rf"\b{k}\b", t):
            return corpus
    m = re.search(r"\b(i[' ]?m|i am)\s+(christian|muslim|buddhist|islam|buddhism|christianity)\b", t)
    if m:
        return RELIGION_MAP.get(m.group(2), None)
    return None

# ========= Sticky note extraction (simple) =========
STICKY_PATTS = [
    r"\bremember that (.+)",
    r"\bplease remember (.+)",
    r"\bnote that (.+)",
    r"\bkeep in mind that (.+)",
]

def extract_sticky_notes(text: str) -> List[str]:
    t = text.lower()
    out = []
    for p in STICKY_PATTS:
        m = re.search(p, t)
        if m:
            val = m.group(1).strip().rstrip(".! ")
            if val:
                out.append(val)
    return out

# ========= Token accounting & memory window =========
def count_tokens(messages: List[Dict[str, str]], model: str) -> int:
    try:
        enc = tiktoken.encoding_for_model(model)
    except Exception:
        enc = tiktoken.get_encoding("cl100k_base")
    total = 0
    # rough accounting for ChatML-like format
    for m in messages:
        total += 4  # role + overhead
        total += len(enc.encode(m.get("content") or ""))
    total += 2
    return total

def trim_history_for_budget(
    base_msgs: List[Dict[str, str]],
    summary: str,
    history: List[Dict[str, str]],
    new_user: Dict[str, str],
    model: str,
    max_tokens: int = 7000,
    reserve_for_output: int = 600,
) -> Tuple[List[Dict[str,str]], str, List[Dict[str,str]]]:
    """
    Build a message list fitting within token budget:
    [system, summary?, sticky?, ...history_tail..., new_user]
    Returns (messages_without_new_user, summary_text, history_tail)
    """
    msgs: List[Dict[str, str]] = base_msgs.copy()

    if summary:
        msgs.append({"role": "system", "content": f"Conversation Summary:\n{summary}"})

    # we add history tail from the end until budget nearly hit
    tail: List[Dict[str, str]] = []
    for m in reversed(history):
        candidate = msgs + list(reversed(tail)) + [m, new_user]
        if count_tokens(candidate, model) < (max_tokens - reserve_for_output):
            tail.insert(0, m)
        else:
            break

    return msgs + tail, summary, tail

def summarize_past(client: OpenAI, model: str, summary: str, overflow: List[Dict[str,str]]) -> str:
    """
    Compress older messages into an updated summary.
    """
    # build text of overflow
    text = ""
    for m in overflow:
        prefix = "User" if m["role"] == "user" else "Assistant"
        text += f"{prefix}: {m['content']}\n"
    prompt = (
        "Summarize the following prior dialogue into crisp bullets with concrete facts/preferences, "
        "so a chatbot can remember them for future turns. Keep under 120 words. "
        "If there are user-stated facts to remember (e.g., 'the penguin is painted blue'), include them verbatim.\n\n"
        f"Existing summary (may be empty):\n{summary or '(none)'}\n\n"
        f"New dialogue to fold in:\n{text}"
    )
    try:
        comp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You compress chat history into compact factual summaries."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            max_tokens=180,
        )
        return comp.choices[0].message.content.strip()
    except Exception:
        # fallback: append raw lines truncated
        joined = text.strip().splitlines()
        return (summary + "\n" + "\n".join(joined[-8:])).strip()

# ========= RAG load (multi-corpus) =========
BASE_DIR = Path(__file__).resolve().parent
RAG_DIR = BASE_DIR / "rag_data"
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

# ========= Retrieval helpers =========
JUNK_PATTERNS = [
    r"Downloaded from www\.holybooks\.com",
    r"^Page \d+",
    r"^\s*The Book of .*",
]
JUNK_RE = re.compile("|".join(JUNK_PATTERNS), re.IGNORECASE | re.MULTILINE)

KEYWORD_HINTS = {
    "courage": ["courage", "be strong", "fear not", "dismayed", "valour", "bold"],
    "strength": ["strength", "mighty", "power", "fortify", "be strong"],
    "patience": ["patience", "patient", "steadfast", "sabr", "persevere"],
    "hope": ["hope", "mercy", "deliver", "trust", "faithfulness"],
}

def want_verse(text: str) -> bool:
    t = text.lower()
    return any(w in t for w in ["verse", "scripture", "quote", "passage", "ayah"])

def hint_terms(text: str) -> List[str]:
    t = text.lower()
    bag = []
    for k, words in KEYWORD_HINTS.items():
        if k in t:
            bag.extend(words)
    return bag

def _filter_hits(hits: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    keep = []
    for h in hits:
        txt = h["text"]
        if JUNK_RE.search(txt):
            continue
        if len(txt.strip()) < 60:
            continue
        keep.append(h)
    return keep

def retrieve_tfidf(query: str, corpus: str, k: int = 5, extra_terms: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    c = CORPORA.get(corpus)
    if not c:
        return []
    tfidf: TfidfVectorizer = c["tfidf"]
    tfidf_mat = c["tfidf_mat"]
    docs = c["docs"]

    q = query
    if extra_terms:
        q += " " + " ".join(extra_terms)

    qv = tfidf.transform([q])                # (1, V)
    sims = (tfidf_mat @ qv.T).toarray().ravel()
    top_idx = sims.argsort()[::-1][:max(k*3, 10)]
    hits = []
    for i in top_idx:
        hits.append({
            "score": float(sims[i]),
            "id": docs[i]["id"],
            "page": docs[i].get("page"),
            "text": docs[i]["text"],
            "corpus": corpus,
        })
    hits = _filter_hits(hits)[:k]
    return hits

def retrieve_best_across(query: str, k: int = 5, extra_terms: Optional[List[str]] = None):
    best: List[Dict[str, Any]] = []
    best_name = None
    best_topscore = -1.0
    for name in CORPORA.keys():
        hits = retrieve_tfidf(query, name, k=k, extra_terms=extra_terms)
        if not hits:
            continue
        if hits[0]["score"] > best_topscore:
            best_topscore = hits[0]["score"]
            best = hits
            best_name = name
    return best_name, best

def build_sources_block(hits: List[Dict[str, Any]]) -> str:
    if not hits:
        return ""
    lines = []
    for i, h in enumerate(hits, start=1):
        label = f"[S{i}]"
        page = f"p{h['page']}" if h.get("page") else ""
        lines.append(f"{label} ({page} {h['id']}) {h['text']}")
    return "\n\n".join(lines)

def sources_event(hits: List[Dict[str, Any]], selected_corpus: Optional[str]) -> str:
    payload = {"sources": hits, "selected_corpus": selected_corpus}
    return json.dumps(payload, ensure_ascii=False)

# ========= Health =========
@app.get("/")
def root():
    return {"ok": True, "message": "API is running", "endpoints": ["/chat_sse", "/chat_rag_json", "/diag", "/diag_rag"]}

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

# ========= SSE chat (with memory) =========
@app.post("/chat_sse")
async def chat_sse(request: Request):
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "Invalid JSON"}, status_code=400)

    user_msg: str = (body.get("message") or "").strip()
    if not user_msg:
        return JSONResponse({"error": "Missing 'message'."}, status_code=400)

    sid = get_or_create_sid(request)
    sess = SESSIONS[sid]

    # Update religion if the user hints it
    detected = detect_religion(user_msg)
    if detected:
        sess["religion"] = detected

    # Capture sticky notes (optional but helpful)
    notes = extract_sticky_notes(user_msg)
    if notes:
        for n in notes:
            if n not in sess["sticky"]:
                sess["sticky"].append(n)

    # Retrieval decision
    model = pick_model()
    client = get_client()

    extra = hint_terms(user_msg)
    want_scripture = want_verse(user_msg)

    selected_corpus: Optional[str] = sess.get("religion")
    hits: List[Dict[str, Any]] = []

    clarify_needed = False
    auto_note = None

    if selected_corpus:
        if want_scripture:
            hits = retrieve_tfidf(user_msg, selected_corpus, k=4, extra_terms=extra)
    else:
        if want_scripture:
            name, best = retrieve_best_across(user_msg, k=4, extra_terms=extra)
            if name and best and best[0]["score"] >= 0.28:
                selected_corpus = name
                hits = best
                auto_note = f"(Since you haven’t told me your tradition yet, I’m sharing a fitting verse from the {name.replace('_', ' ')}. You can say “Bible”, “Qur’an”, or “Suttas” to switch.)"
            else:
                clarify_needed = True

    # Build Sources block (only if hits)
    source_block = build_sources_block(hits) if hits else ""

    # Build current user turn (augmented if sources)
    augmented_user = user_msg + ("\n\n---\nSources:\n" + source_block if source_block else "")

    # ===== Build message list with MEMORY =====
    base_msgs = [{"role": "system", "content": SYSTEM_PROMPT}]

    # Sticky are high-priority facts
    if sess.get("sticky"):
        base_msgs.append({"role": "system", "content": "Sticky Notes (persist across this session):\n- " + "\n- ".join(sess["sticky"])})

    # Decide the "new user turn" content (clarify or normal)
    new_user_content = (
        f"{user_msg}\n\n—\nBefore I pull scripture: which tradition should I draw from—Bible, Qur’an, or the Sutta Pitaka?"
        if clarify_needed else
        (auto_note + "\n\n" + augmented_user if auto_note else augmented_user)
    )
    new_user = {"role": "user", "content": new_user_content}

    # Trim history to fit budget
    msgs_wo_new, summary_text, history_tail = trim_history_for_budget(
        base_msgs, sess.get("summary", ""), sess.get("history", []), new_user, model
    )
    messages = msgs_wo_new + [new_user]

    # ===== Stream response and record assistant text into history =====
    async def token_stream():
        yield ": connected\n\n"
        buf = []
        try:
            with client.chat.completions.stream(
                model=model,
                messages=messages,
                temperature=0.4,
                max_tokens=450,
            ) as stream:
                for event in stream:
                    if event.type == "token":
                        tok = event.token
                        buf.append(tok)
                        yield f"data: {tok}\n\n"
                    elif event.type == "completed":
                        break
                yield "data: [DONE]\n\n"
        except Exception as e:
            err = str(e)
            yield f"data: [ERROR] {err}\n\n"

        # Emit sources after text if we used RAG
        if hits:
            yield "event: sources\n"
            yield f"data: {sources_event(hits, selected_corpus)}\n\n"

        # ====== Update memory (history + summary) ======
        try:
            # Append the turn we just handled
            # 1) user
            sess["history"].append({"role": "user", "content": new_user_content})
            # 2) assistant (concatenate streamed tokens)
            assistant_text = "".join(buf).strip()
            sess["history"].append({"role": "assistant", "content": assistant_text})

            # Keep recent tail length reasonable
            # If history grows large, fold older part into summary
            if len(sess["history"]) > 20:
                # overflow = everything except last 10 messages
                overflow = sess["history"][:-10]
                sess["history"] = sess["history"][-10:]
                sess["summary"] = summarize_past(client, model, sess.get("summary", ""), overflow)

        except Exception:
            # Never crash the stream for memory bookkeeping issues
            pass

    resp = StreamingResponse(token_stream(), media_type="text/event-stream")
    resp.set_cookie("sid", sid, httponly=True, samesite="Lax")
    return resp

# ========= Non-stream JSON (debug) =========
@app.post("/chat_rag_json")
def chat_rag_json(body: Dict[str, Any], request: Request = None):
    try:
        user_msg: str = (body.get("message") or "").strip()
        chosen: Optional[str] = body.get("religion")
        if not user_msg:
            return JSONResponse({"error": "Missing 'message'."}, status_code=400)

        # Optional: include session memory here, too
        # If request exists, honor its cookie session
        sid = None
        sess = {"history": [], "summary": "", "sticky": []}
        if request:
            sid = request.cookies.get("sid")
            if sid and sid in SESSIONS:
                sess = SESSIONS[sid]

        extra = hint_terms(user_msg)
        if chosen:
            if chosen not in CORPORA:
                return JSONResponse({"error": f"Unknown religion '{chosen}'."}, status_code=400)
            hits = retrieve_tfidf(user_msg, chosen, k=4, extra_terms=extra)
            selected = chosen
        else:
            selected, hits = retrieve_best_across(user_msg, k=4, extra_terms=extra)

        source_block = build_sources_block(hits) if hits else ""
        model = pick_model()
        client = get_client()

        sys = SYSTEM_PROMPT
        msgs = [{"role": "system", "content": sys}]
        if sess.get("sticky"):
            msgs.append({"role": "system", "content": "Sticky Notes:\n- " + "\n- ".join(sess["sticky"])})
        if sess.get("summary"):
            msgs.append({"role": "system", "content": "Conversation Summary:\n" + sess["summary"]})
        # use tail of history (without overcomplicating tokens here)
        for m in sess.get("history", [])[-8:]:
            msgs.append(m)

        user = user_msg + ("\n\n---\nSources:\n" + source_block if source_block else "")
        msgs.append({"role": "user", "content": user})

        comp = client.chat.completions.create(
            model=model,
            messages=msgs,
            temperature=0.4,
            max_tokens=450,
        )
        text = comp.choices[0].message.content
        return {"response": text, "sources": hits, "selected_corpus": selected}
    except Exception as e:
        if str(e).startswith("test-") or "model_not_found" in str(e):
            return JSONResponse(
                {"error": "Invalid OPENAI_MODEL. Set a real model (e.g., gpt-4o-mini) or unset the variable."},
                status_code=500,
            )
        return JSONResponse({"error": str(e)}, status_code=500)
