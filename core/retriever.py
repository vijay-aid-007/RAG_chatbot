"""
core/retriever.py
=================
RAG retrieval layer.  Given a natural-language query, returns:

    {
        "topic_summaries" : [{"topic_id", "summary", "score"}, ...],
        "chunks"          : [{"chunk_id", "text", "score", ...}, ...],
        "combined_context": str,   # ready to paste into LLM prompt
    }

Two-stage retrieval:
    Stage 1  → embed topic summaries in memory, cosine-rank, top-K_TOPICS
    Stage 2  → FAISS search over raw message chunks, top-K_CHUNKS
    Stage 3  → deduplicate & merge into `combined_context`

Topic summaries are embedded lazily (on first query, then cached in RAM).
"""

from __future__ import annotations
import json
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import TOP_K_TOPICS, TOP_K_CHUNKS, TOPIC_CHECKPOINTS, MSG_CHECKPOINTS
from core.embedder import embed, batch_cosine_similarity
from core.vector_store import search as chunk_search


# ── Topic summary retrieval ───────────────────────────────────────────────────

_topic_cache: dict = {}   # {"summaries": [...], "matrix": np.ndarray}


def _load_topic_embeddings() -> None:
    global _topic_cache
    if _topic_cache:
        return  # already loaded

    if not TOPIC_CHECKPOINTS.exists():
        raise FileNotFoundError(
            "Topic checkpoints not found. Run build_pipeline.py first."
        )

    with open(TOPIC_CHECKPOINTS, encoding="utf-8") as fh:
        topics = json.load(fh)

    summaries = [t for t in topics if t.get("summary")]
    texts = [t["summary"] for t in summaries]

    print(f"[retriever] Embedding {len(texts)} topic summaries…")
    matrix = embed(texts)   # (N, 384)

    _topic_cache["summaries"] = summaries
    _topic_cache["matrix"] = matrix
    print("[retriever] Topic embeddings ready.")


def _retrieve_topic_summaries(query_text: str, top_k: int = TOP_K_TOPICS) -> list[dict]:
    _load_topic_embeddings()
    summaries = _topic_cache["summaries"]
    matrix    = _topic_cache["matrix"]

    q_vec = embed(query_text)[0]
    scores = batch_cosine_similarity(q_vec, matrix)

    top_indices = np.argsort(scores)[::-1][:top_k]
    results = []
    for idx in top_indices:
        t = dict(summaries[idx])
        t["score"] = float(scores[idx])
        results.append(t)
    return results


# ── Public API ────────────────────────────────────────────────────────────────

def retrieve(query_text: str) -> dict:
    """
    Full RAG retrieval for a query.

    Returns:
        {
            "topic_summaries" : list of topic dicts with scores,
            "chunks"          : list of chunk dicts with scores,
            "combined_context": str (formatted context for LLM),
        }
    """
    topic_results = _retrieve_topic_summaries(query_text, top_k=TOP_K_TOPICS)
    chunk_results = chunk_search(query_text, top_k=TOP_K_CHUNKS)

    # Build combined context string for LLM
    context_parts = []

    context_parts.append("=== RELEVANT TOPIC SUMMARIES ===")
    for t in topic_results:
        context_parts.append(
            f"[Topic {t['topic_id']} | msgs {t['start_msg_id']}–{t['end_msg_id']} "
            f"| sim={t['score']:.3f}]\n{t['summary']}"
        )

    context_parts.append("\n=== RELEVANT MESSAGE CHUNKS ===")
    for c in chunk_results:
        context_parts.append(
            f"[Chunk: msgs {c['start_msg_id']}–{c['end_msg_id']} | sim={c['score']:.3f}]\n"
            f"{c['text']}"
        )

    combined_context = "\n\n".join(context_parts)

    return {
        "topic_summaries" : topic_results,
        "chunks"          : chunk_results,
        "combined_context": combined_context,
    }


if __name__ == "__main__":
    result = retrieve("What are this person's hobbies?")
    print(result["combined_context"][:800])
