"""
core/chatbot.py
===============
Part 3 — Chatbot that uses both the RAG system (Part 1) and
the persona (Part 2) to answer natural-language questions.

Public function:
    answer(user_question: str) -> str

The bot:
    1. Decides if the query is persona-focused or conversation-focused.
    2. Retrieves relevant context from the RAG system.
    3. Loads the persona JSON.
    4. Constructs a well-structured prompt and calls Groq LLM.
    5. Returns a plain-text answer.
"""

from __future__ import annotations
import json
from pathlib import Path
from groq import Groq

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import GROQ_API_KEY, LLM_MODEL, LLM_MAX_TOKENS
from core.retriever import retrieve
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


# ── Intent classifier ─────────────────────────────────────────────────────────

_PERSONA_KEYWORDS = {
    "person", "who", "habit", "hobby", "personality", "trait",
    "talk", "speak", "communicate", "emoji", "tone", "style",
    "kind of", "type of", "character", "feel", "emotion",
    "relationship", "family", "work", "job", "career", "sleep",
    "food", "diet", "pet", "interest", "like", "love", "enjoy",
}

def _is_persona_query(question: str) -> bool:
    """True if the question seems persona/user-profile oriented."""
    lower = question.lower()
    return any(kw in lower for kw in _PERSONA_KEYWORDS)


# ── Core answer function ──────────────────────────────────────────────────────

def answer(user_question: str, verbose: bool = False) -> str:
    """
    Generate an answer to `user_question` using RAG + persona.

    Args:
        user_question : natural language question from the user
        verbose       : if True, print retrieved context to stdout

    Returns:
        Plain-text answer string.
    """
    # 1. Retrieve RAG context
    retrieval = retrieve(user_question)
    rag_context = retrieval["combined_context"]

    if verbose:
        print("\n── RAG Context ──────────────────────────────")
        print(rag_context[:1200])
        print("─────────────────────────────────────────────\n")

    # 2. Load persona
    try:
        persona = load_persona()
        persona_str = json.dumps(persona, indent=2, ensure_ascii=False)
    except FileNotFoundError:
        persona_str = "Persona not available. Run the build pipeline first."

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