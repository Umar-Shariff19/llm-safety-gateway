# app/main.py
import os
from typing import List

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

from dotenv import load_dotenv
from groq import Groq  # Groq SDK

from app.safety_engine import analyze_prompt

# -----------------------------
# Load environment variables
# -----------------------------
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Groq client
groq_client = Groq(api_key=GROQ_API_KEY)

app = FastAPI(title="LLM Safety Gateway - Groq Edition")

# -----------------------------
# Serve static frontend files
# -----------------------------
app.mount("/static", StaticFiles(directory="app/static"), name="static")


# -----------------------------
# Models
# -----------------------------
class PromptRequest(BaseModel):
    prompt: str
    history: List[str] = []


# -----------------------------
# Root → serve UI
# -----------------------------
@app.get("/")
async def serve_index():
    return FileResponse("app/static/index.html")


# -----------------------------
# /analyze (safety engine only)
# -----------------------------
@app.post("/analyze")
async def analyze(req: PromptRequest):
    return analyze_prompt(req.prompt, req.history)


# -----------------------------
# Groq LLM call helper
# -----------------------------
async def call_groq(prompt: str) -> str:
    if not GROQ_API_KEY:
        return "(Error: GROQ_API_KEY missing in .env)"

    try:
        response = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ],
        )

        # FIXED: access content as an attribute, not dict
        return response.choices[0].message.content

    except Exception as e:
        return f"(LLM Error: {str(e)})"


# -----------------------------
# /simulate (Safety + Groq LLM)
# -----------------------------
@app.post("/simulate")
async def simulate(req: PromptRequest):
    """
    1. Safety Gateway: drift, fragmentation, sanitization
    2. If allowed → Groq LLaMA 3.1 LLM call
    3. Returns: safety_output + llm_output
    """

    # Step 1 — safety engine
    safety = analyze_prompt(req.prompt, req.history)
    final_prompt = safety["sanitized_prompt"]

    # Blocked output (default)
    llm_output = (
        "⚠️ Safety Gateway blocked or sanitized this prompt.\n"
        f"Safe version: {final_prompt}"
    )

    # Step 2 — only call Groq when allowed
    if safety["action"] == "allow":
        llm_output = await call_groq(final_prompt)

    return {
        "safety_output": safety,
        "llm_output": llm_output,
    }
