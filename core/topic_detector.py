"""
core/topic_detector.py
======================
Detects topic boundaries in a chronologically ordered message stream.

Algorithm (sliding window cosine):
────────────────────────────────────────────────────────────────────
1.  Group messages into overlapping windows of size W.
2.  Embed the concatenated text of each window.
3.  Compute cosine similarity between consecutive window embeddings.
4.  When similarity drops BELOW `threshold` AND the current segment is
    at least `min_segment_len` messages long → mark a topic boundary.

This is a lightweight, interpretable approach that requires zero training
data and runs fully offline.  It out-performs naive single-message
comparison because the window smooths out one-off noisy utterances.

Output — list of TopicSegment dicts:
{
    "topic_id"    : int,          # 1-based
    "start_msg_id": int,
    "end_msg_id"  : int,
    "messages"    : [msg_dict, ...],
    "summary"     : str | None,   # filled in by checkpointer
}
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import List

import numpy as np
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    TOPIC_WINDOW_SIZE, TOPIC_SIMILARITY_THRESH, TOPIC_MIN_SEGMENT_LEN,
    TOPIC_CHECKPOINTS,
)
from core.embedder import embed, cosine_similarity


# ── helpers ──────────────────────────────────────────────────────────────────

def _window_text(messages: list[dict], start: int, size: int) -> str:
    """Concatenate `size` messages starting at index `start`."""
    window = messages[start : start + size]
    return " ".join(m["text"] for m in window)


# ── Core detection ────────────────────────────────────────────────────────────

def detect_topic_boundaries(messages: list[dict]) -> list[int]:
    """
    Return a sorted list of message *indices* (0-based into `messages`)
    where a new topic begins.  Index 0 is always included.

    Strategy:
        - Slide a window of size W across the message list.
        - Embed each window.
        - A boundary fires when sim(window[i], window[i+1]) < threshold
          AND we have accumulated at least min_segment_len messages since
          the last boundary.
    """
    W         = TOPIC_WINDOW_SIZE
    threshold = TOPIC_SIMILARITY_THRESH
    min_len   = TOPIC_MIN_SEGMENT_LEN
    n         = len(messages)

    boundaries: list[int] = [0]   # first topic always starts at 0
    last_boundary = 0

    # Build window embeddings in one batch for efficiency
    indices = list(range(0, n - W + 1))  # valid window start positions
    if not indices:
        return boundaries  # too few messages

    print(f"[topic_detector] Embedding {len(indices)} windows (W={W})…")
    texts = [_window_text(messages, i, W) for i in indices]

    # Encode in batches to avoid OOM
    BATCH = 512
    all_vecs: list[np.ndarray] = []
    for start in tqdm(range(0, len(texts), BATCH), desc="Embedding windows"):
        batch_vecs = embed(texts[start : start + BATCH])
        all_vecs.append(batch_vecs)
    vecs = np.vstack(all_vecs)   # shape (len(indices), 384)

    print("[topic_detector] Scanning for boundaries…")
    for i in range(len(indices) - 1):
        sim = cosine_similarity(vecs[i], vecs[i + 1])
        msg_idx = indices[i + 1]   # message index where the new window starts

        if (sim < threshold) and (msg_idx - last_boundary >= min_len):
            boundaries.append(msg_idx)
            last_boundary = msg_idx

    print(f"[topic_detector] Found {len(boundaries)} topic segments.")
    return sorted(set(boundaries))


def build_topic_segments(
    messages: list[dict],
    boundaries: list[int],
) -> list[dict]:
    """
    Convert boundary indices into TopicSegment dicts (without summaries yet).
    """
    segments = []
    for topic_id, (start_idx, end_idx) in enumerate(
        zip(boundaries, boundaries[1:] + [len(messages)]),
        start=1,
    ):
        seg_msgs = messages[start_idx:end_idx]
        segments.append({
            "topic_id"    : topic_id,
            "start_msg_id": seg_msgs[0]["msg_id"],
            "end_msg_id"  : seg_msgs[-1]["msg_id"],
            "messages"    : seg_msgs,
            "summary"     : None,
        })
    return segments


# ── Persistence ───────────────────────────────────────────────────────────────

def save_topic_segments(segments: list[dict], path: Path = TOPIC_CHECKPOINTS) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Don't store raw messages (too large) — store only metadata + summary
    slim = [
        {
            "topic_id"    : s["topic_id"],
            "start_msg_id": s["start_msg_id"],
            "end_msg_id"  : s["end_msg_id"],
            "msg_count"   : len(s["messages"]),
            "summary"     : s["summary"],
        }
        for s in segments
    ]
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(slim, fh, indent=2, ensure_ascii=False)
    print(f"[topic_detector] Saved {len(slim)} topic checkpoints → {path}")


def load_topic_segments(path: Path = TOPIC_CHECKPOINTS) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


if __name__ == "__main__":
    from core.parser import get_messages
    msgs = get_messages()
    boundaries = detect_topic_boundaries(msgs)
    segs = build_topic_segments(msgs, boundaries)
    print(f"Segment 1: msg {segs[0]['start_msg_id']}–{segs[0]['end_msg_id']}")
    print(f"Total segments: {len(segs)}")
