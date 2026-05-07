"""
core/chatbot.py
===============
Part 3 — Chatbot that uses both the RAG system (Part 1) and
the persona (Part 2) to answer natural-language questions.

Public function:
    answer(user_question: str) -> str
"""

from __future__ import annotations
import json
from pathlib import Path
from groq import Groq

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import GROQ_API_KEY, LLM_MODEL, LLM_MAX_TOKENS, TOPIC_CHECKPOINTS
from core.persona_extractor import load_persona


# ── Prompt templates ──────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an AI assistant that has deeply analysed a large dataset of conversations.
You have access to:
1. A structured persona extracted from those conversations (habits, personality, communication style, facts).
2. Relevant conversation excerpts and topic summaries retrieved for the user's question.

Answer the user's question accurately and concisely.
Base your answer ONLY on the provided context — do not invent details.
If the context does not contain enough information, say so honestly.
"""

ANSWER_PROMPT = """
## User Persona
{persona_json}

## Retrieved Context
{rag_context}

## User Question
{question}

## Your Answer
"""


# ── Safe retrieval (handles missing FAISS gracefully) ─────────────────────────

def _safe_retrieve(question: str) -> str:
    """
    Try full RAG retrieval. If FAISS index is missing,
    fall back to topic summaries + persona only.
    """
    # Try full retrieval first
    try:
        from core.retriever import retrieve
        result = retrieve(question)
        return result["combined_context"]
    except Exception as faiss_err:
        if "FAISS" in str(faiss_err) or "faiss" in str(faiss_err) or "not found" in str(faiss_err):
            # Fallback: use topic summaries only
            return _retrieve_topics_only(question)
        raise


def _retrieve_topics_only(question: str) -> str:
    """
    Fallback retrieval using only topic checkpoints (no FAISS needed).
    Returns top matching topic summaries based on keyword overlap.
    """
    try:
        if not TOPIC_CHECKPOINTS.exists():
            return "No context available."

        with open(TOPIC_CHECKPOINTS, encoding="utf-8") as fh:
            topics = json.load(fh)

        # Simple keyword-based scoring
        question_words = set(question.lower().split())
        scored = []
        for t in topics:
            summary = t.get("summary", "")
            if not summary:
                continue
            summary_words = set(summary.lower().split())
            score = len(question_words & summary_words)
            scored.append((score, t))

        # Sort by score and take top 5
        scored.sort(key=lambda x: x[0], reverse=True)
        top = scored[:5]

        context_parts = ["=== RELEVANT TOPIC SUMMARIES ==="]
        for score, t in top:
            context_parts.append(
                f"[Topic {t.get('topic_id', '?')} | "
                f"msgs {t.get('start_msg_id', '?')}–{t.get('end_msg_id', '?')}]\n"
                f"{t['summary']}"
            )

        return "\n\n".join(context_parts)

    except Exception as e:
        return f"Context retrieval failed: {e}"


# ── Core answer function ───────────────────────────────────────────────────────

def answer(user_question: str, verbose: bool = False) -> str:
    """
    Generate an answer to `user_question` using RAG + persona.
    Falls back gracefully if FAISS index is unavailable.
    """
    # 1. Retrieve context (with fallback)
    rag_context = _safe_retrieve(user_question)

    if verbose:
        print("\n── RAG Context ──────────────────────────────")
        print(rag_context[:1200])
        print("─────────────────────────────────────────────\n")

    # 2. Load persona
    try:
        persona = load_persona()
        persona_str = json.dumps(persona, indent=2, ensure_ascii=False)
    except FileNotFoundError:
        persona_str = "Persona not available."

    # 3. Build prompt
    prompt = ANSWER_PROMPT.format(
        persona_json=persona_str,
        rag_context=rag_context,
        question=user_question,
    )

    # 4. Call Groq LLM
    if not GROQ_API_KEY:
        return (
            "[Error] GROQ_API_KEY not set. "
            "Add it to your .env file and restart."
        )

    try:
        client = Groq(api_key=GROQ_API_KEY)
        response = client.chat.completions.create(
            model=LLM_MODEL,
            max_tokens=LLM_MAX_TOKENS,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": prompt},
            ],
        )
        return response.choices[0].message.content.strip()
    except Exception as exc:
        return f"[Error calling Groq LLM] {exc}"


if __name__ == "__main__":
    questions = [
        "What kind of person is this user?",
        "What are their habits?",
        "How do they talk?",
        "What topics come up most in their conversations?",
    ]
    for q in questions:
        print(f"\nQ: {q}")
        print(f"A: {answer(q)}\n")
        print("=" * 60)