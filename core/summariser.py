"""
core/summariser.py
==================
Generates summaries for topic segments and 100-message checkpoints
using the Groq API.

Two public functions:
    summarise_topic_segment(messages)      → str
    summarise_message_window(messages)     → str

Fallback: if the API call fails, returns a simple extractive summary
so the pipeline never crashes due to API errors.
"""

from __future__ import annotations
from groq import Groq
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    GROQ_API_KEY, LLM_MODEL, LLM_MAX_TOKENS,
    TOPIC_SUMMARY_PROMPT, MSG100_SUMMARY_PROMPT,
)


# ── Extractive fallback ───────────────────────────────────────────────────────

def _extractive_fallback(messages: list[dict], n_sentences: int = 5) -> str:
    seen = set()
    parts = []
    for msg in messages:
        t = msg["text"].strip()
        if t and t not in seen:
            seen.add(t)
            parts.append(t)
        if len(parts) >= n_sentences:
            break
    return " | ".join(parts)


# ── LLM call ─────────────────────────────────────────────────────────────────

def _llm_summarise(prompt_template: str, messages: list[dict]) -> str:
    segment_text = "\n".join(
        f"[{m['speaker']}]: {m['text']}" for m in messages
    )
    prompt = prompt_template.format(segment=segment_text, messages=segment_text)

    try:
        client = Groq(api_key=GROQ_API_KEY)
        response = client.chat.completions.create(
            model=LLM_MODEL,
            max_tokens=LLM_MAX_TOKENS,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content.strip()
    except Exception as exc:
        print(f"[summariser] LLM call failed ({exc}), using extractive fallback.")
        return _extractive_fallback(messages)


# ── Public API ────────────────────────────────────────────────────────────────

def summarise_topic_segment(messages: list[dict]) -> str:
    if len(messages) > 200:
        sampled = (
            messages[:20]
            + messages[len(messages)//2 - 10 : len(messages)//2 + 10]
            + messages[-20:]
        )
        sampled = list({m["msg_id"]: m for m in sampled}.values())
        sampled.sort(key=lambda m: m["msg_id"])
        messages = sampled[:60]
    return _llm_summarise(TOPIC_SUMMARY_PROMPT, messages)


def summarise_message_window(messages: list[dict]) -> str:
    return _llm_summarise(MSG100_SUMMARY_PROMPT, messages)


if __name__ == "__main__":
    fake_msgs = [
        {"msg_id": i, "speaker": "User 1" if i % 2 else "User 2", "text": f"Test message {i}"}
        for i in range(1, 11)
    ]
    print(summarise_topic_segment(fake_msgs))