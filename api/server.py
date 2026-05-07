"""
api/server.py
=============
FastAPI REST API — alternative to the Gradio UI.
Useful if you want to build a custom frontend or integrate with other tools.

Endpoints:
    POST /chat        — answer a question
    GET  /persona     — return the full persona JSON
    GET  /topics      — return all topic checkpoints
    GET  /health      — system status

Run:
    uvicorn api.server:app --host 0.0.0.0 --port 8000 --reload
"""

from __future__ import annotations
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.chatbot import answer
from core.persona_extractor import load_persona
from core.topic_detector import load_topic_segments
from config import PERSONA_JSON, TOPIC_CHECKPOINTS, FAISS_INDEX_PATH


# ── App setup ─────────────────────────────────────────────────────────────────

app = FastAPI(
    title="RAG Conversation Chatbot API",
    description=(
        "Query a RAG system built over conversation data. "
        "Combines topic checkpoints, message chunk retrieval, and persona extraction."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Schemas ───────────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    question: str

class ChatResponse(BaseModel):
    question: str
    answer  : str


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health() -> dict:
    return {
        "status"   : "ok",
        "persona"  : PERSONA_JSON.exists(),
        "topics"   : TOPIC_CHECKPOINTS.exists(),
        "faiss"    : FAISS_INDEX_PATH.exists(),
    }


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    reply = answer(req.question)
    return ChatResponse(question=req.question, answer=reply)


@app.get("/persona")
def persona() -> dict:
    try:
        return load_persona()
    except FileNotFoundError:
        raise HTTPException(
            status_code=503,
            detail="Persona not built. Run build_pipeline.py first."
        )


@app.get("/topics")
def topics() -> list[dict]:
    if not TOPIC_CHECKPOINTS.exists():
        raise HTTPException(
            status_code=503,
            detail="Topic checkpoints not built. Run build_pipeline.py first."
        )
    return load_topic_segments()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.server:app", host="0.0.0.0", port=8000, reload=True)
