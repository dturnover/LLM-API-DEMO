import os
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
def get_client():
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY not set")
    return OpenAI(api_key=key)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten in prod
    allow_methods=["GET","POST","OPTIONS"],
    allow_headers=["Content-Type","Authorization"],
)

SYSTEM_PROMPT = (
    "You are <BrandName>'s website assistant. Be concise, friendly, on-brand. "
    "Ask a short clarifying question if the request is ambiguous."
)

@app.get("/")
def root():
    return {"ok": True, "message": "API is running", "endpoints": ["/chat (stream)", "/chat_json (non-stream)"]}

# --- Non-streaming: easy curl testing ---
@app.post("/chat_json")
async def chat_json(request: Request):
    body = await request.json()
    user_message = (body.get("message") or "").strip()
    if not user_message:
        return JSONResponse({"error":"message is required"}, status_code=400)

    client = get_client()
    resp = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role":"system","content": SYSTEM_PROMPT},
            {"role":"user","content": user_message},
        ],
        stream=False,
    )
    content = resp.choices[0].message.content
    return {"response": content}

# --- Streaming: for the web UI ---
@app.post("/chat")
async def chat_stream(request: Request):
    body = await request.json()
    user_message = (body.get("message") or "").strip()
    if not user_message:
        return JSONResponse({"error":"message is required"}, status_code=400)

    def gen():

        client = get_client()
        stream = client.chat.completions.create(
            model="gpt-4o",
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
        # explicitly end the stream so curl doesn't complain
        yield ""

    return StreamingResponse(gen(), media_type="text/plain")

