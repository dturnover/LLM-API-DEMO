import os, traceback
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from openai import OpenAI

def get_client():
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY not set")
    return OpenAI(api_key=key)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten later
    allow_methods=["GET","POST","OPTIONS"],
    allow_headers=["Content-Type","Authorization"],
)

SYSTEM_PROMPT = (
    "You are <BrandName>'s website assistant. Be concise and friendly. "
    "If the request is ambiguous, ask a brief clarifying question."
)

@app.get("/")
def root():
    return {"ok": True, "message": "API is running", "endpoints": ["/chat (stream)", "/chat_json (non-stream)", "/diag"]}

@app.get("/diag")
def diag():
    try:
        client = get_client()
        # quick call that fails fast if key/billing is wrong
        _ = client.models.list()
        return {"ok": True}
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)

@app.post("/chat_json")
async def chat_json(request: Request):
    try:
        body = await request.json()
        user_message = (body.get("message") or "").strip()
        if not user_message:
            return JSONResponse({"error": "message is required"}, status_code=400)

        client = get_client()
        resp = client.chat.completions.create(
            # use mini first to avoid permission issues; swap to gpt-4o after
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
                    {"role":"system","content": SYSTEM_PROMPT},
                    {"role":"user","content": user_message},
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
