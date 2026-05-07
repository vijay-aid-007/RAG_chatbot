"""
core/vector_store.py
====================
Builds and queries a FAISS flat-IP index over message chunks.

Each indexed unit ("chunk") = CHUNK_SIZE consecutive messages concatenated.
Metadata (msg_id range, raw text) is stored in a parallel JSONL file.

Architecture:
    - FAISS IndexFlatIP  → exact inner-product search (= cosine on normalised vecs)
    - faiss_meta.jsonl   → one JSON line per indexed chunk
    - Both files are saved/loaded so indexing runs only once.

Public API:
    build_index(messages)                     → None (saves to disk)
    search(query_text, top_k) → list[dict]    → top-k chunk dicts
"""

from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import faiss
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    CHUNK_SIZE,
    FAISS_INDEX_PATH,
    FAISS_META_PATH,
    TOP_K_CHUNKS,
)
from core.embedder import embed


# ── Build ─────────────────────────────────────────────────────────────────────

def _chunk_messages(messages: list[dict], chunk_size: int) -> list[dict]:
    """
    Slide a non-overlapping window of `chunk_size` over the message list.
    Returns list of chunk dicts.
    """
    chunks = []
    for start in range(0, len(messages), chunk_size):
        window = messages[start : start + chunk_size]
        if not window:
            break
        text = "\n".join(f"[{m['speaker']}]: {m['text']}" for m in window)
        chunks.append({
            "chunk_id"    : len(chunks),
            "start_msg_id": window[0]["msg_id"],
            "end_msg_id"  : window[-1]["msg_id"],
            "text"        : text,
        })
    return chunks


def build_index(
    messages: list[dict],
    force: bool = False,
    chunk_size: int = CHUNK_SIZE,
) -> None:
    """
    Build FAISS index from messages and save to disk.
    Skips if index already exists (unless force=True).
    """
    if not force and FAISS_INDEX_PATH.exists() and FAISS_META_PATH.exists():
        print("[vector_store] Index already exists — skipping build. Pass force=True to rebuild.")
        return

    print(f"[vector_store] Chunking {len(messages)} messages (chunk_size={chunk_size})…")
    chunks = _chunk_messages(messages, chunk_size)
    print(f"[vector_store] Created {len(chunks)} chunks.")

    # Encode all chunks
    EMBED_BATCH = 256
    all_vecs: list[np.ndarray] = []
    for start in tqdm(range(0, len(chunks), EMBED_BATCH), desc="Embedding chunks"):
        batch_texts = [c["text"] for c in chunks[start : start + EMBED_BATCH]]
        vecs = embed(batch_texts)
        all_vecs.append(vecs)

    matrix = np.vstack(all_vecs).astype(np.float32)   # (N, 384)

    # Build FAISS index
    dim = matrix.shape[1]
    index = faiss.IndexFlatIP(dim)   # inner product on L2-normed vecs = cosine sim
    index.add(matrix)

    # Save
    FAISS_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(FAISS_INDEX_PATH))
    print(f"[vector_store] FAISS index saved → {FAISS_INDEX_PATH}  ({index.ntotal} vectors)")

    with open(FAISS_META_PATH, "w", encoding="utf-8") as fh:
        for chunk in chunks:
            fh.write(json.dumps(chunk, ensure_ascii=False) + "\n")
    print(f"[vector_store] Metadata saved → {FAISS_META_PATH}")


# ── Load ──────────────────────────────────────────────────────────────────────

_index: faiss.IndexFlatIP | None = None
_meta: list[dict] | None = None


def _load() -> tuple[faiss.IndexFlatIP, list[dict]]:
    global _index, _meta
    if _index is None:
        if not FAISS_INDEX_PATH.exists():
            raise FileNotFoundError(
                "FAISS index not found. Run `python build_pipeline.py` first."
            )
        _index = faiss.read_index(str(FAISS_INDEX_PATH))
        print(f"[vector_store] Loaded FAISS index ({_index.ntotal} vectors).")

    if _meta is None:
        _meta = []
        with open(FAISS_META_PATH, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    _meta.append(json.loads(line))
        print(f"[vector_store] Loaded {len(_meta)} chunk metadata records.")

    return _index, _meta


# ── Search ────────────────────────────────────────────────────────────────────

def search(query_text: str, top_k: int = TOP_K_CHUNKS) -> list[dict]:
    """
    Retrieve the top-k most relevant chunks for `query_text`.

    Returns list of chunk dicts enriched with `score`:
    {
        "chunk_id"    : int,
        "start_msg_id": int,
        "end_msg_id"  : int,
        "text"        : str,
        "score"       : float,   # cosine similarity
    }
    """
    index, meta = _load()
    q_vec = embed(query_text).astype(np.float32)  # (1, 384)

    scores, indices = index.search(q_vec, top_k)   # both shape (1, top_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:   # FAISS returns -1 for unfilled slots
            continue
        chunk = dict(meta[idx])
        chunk["score"] = float(score)
        results.append(chunk)

    return results


if __name__ == "__main__":
    from core.parser import get_messages
    msgs = get_messages()
    build_index(msgs, force=False)
    hits = search("hiking and outdoor activities")
    for h in hits:
        print(f"[score={h['score']:.3f}] msgs {h['start_msg_id']}–{h['end_msg_id']}: {h['text'][:120]}…")
