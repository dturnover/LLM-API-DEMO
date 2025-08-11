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

origins = [ 
    "https://dturnover.github.io"
]

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
