"""
core/checkpointer.py
====================
Orchestrates ALL checkpoint creation in a single pass:

    1. Topic checkpoints   — via topic_detector + summariser
    2. 100-msg checkpoints — every MSG_CHECKPOINT_INTERVAL messages

Both are written to JSON files in storage/.

This is the MOST IMPORTANT file for Part 1.
Run it ONCE after parsing; results are cached.
"""

from __future__ import annotations
import json
from pathlib import Path
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    MSG_CHECKPOINT_INTERVAL,
    TOPIC_CHECKPOINTS,
    MSG_CHECKPOINTS,
)
from core.topic_detector import (
    detect_topic_boundaries,
    build_topic_segments,
    save_topic_segments,
    load_topic_segments,
)
from core.summariser import summarise_topic_segment, summarise_message_window


# ── 100-message checkpoints ───────────────────────────────────────────────────

def build_msg100_checkpoints(
    messages: list[dict],
    path: Path = MSG_CHECKPOINTS,
    force: bool = False,
) -> list[dict]:
    """
    Every MSG_CHECKPOINT_INTERVAL messages → one summary checkpoint.

    Returns list of:
    {
        "checkpoint_id" : int,   # 1-based
        "start_msg_id"  : int,
        "end_msg_id"    : int,
        "summary"       : str,
    }
    """
    if not force and path.exists():
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        print(f"[checkpointer] Loaded {len(data)} msg-100 checkpoints from cache.")
        return data

    N   = MSG_CHECKPOINT_INTERVAL
    checkpoints = []
    ck_id = 1

    for start in tqdm(range(0, len(messages), N), desc="100-msg checkpoints"):
        window = messages[start : start + N]
        if not window:
            break
        summary = summarise_message_window(window)
        checkpoints.append({
            "checkpoint_id": ck_id,
            "start_msg_id" : window[0]["msg_id"],
            "end_msg_id"   : window[-1]["msg_id"],
            "msg_count"    : len(window),
            "summary"      : summary,
        })
        ck_id += 1

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(checkpoints, fh, indent=2, ensure_ascii=False)
    print(f"[checkpointer] Saved {len(checkpoints)} msg-100 checkpoints → {path}")
    return checkpoints


# ── Topic checkpoints ─────────────────────────────────────────────────────────

def build_topic_checkpoints(
    messages: list[dict],
    force: bool = False,
) -> list[dict]:
    """
    Detect topic boundaries, generate summaries for each segment,
    persist to TOPIC_CHECKPOINTS JSON.

    Returns list of slim topic dicts (no raw messages stored).
    """
    if not force and TOPIC_CHECKPOINTS.exists():
        data = load_topic_segments()
        print(f"[checkpointer] Loaded {len(data)} topic checkpoints from cache.")
        return data

    # Step 1: detect boundaries
    boundaries = detect_topic_boundaries(messages)

    # Step 2: build segments
    segments = build_topic_segments(messages, boundaries)

    # Step 3: generate summaries
    print(f"[checkpointer] Generating summaries for {len(segments)} topic segments…")
    for seg in tqdm(segments, desc="Topic summaries"):
        seg["summary"] = summarise_topic_segment(seg["messages"])

    # Step 4: persist (without raw messages)
    save_topic_segments(segments)

    # Return slim format
    return [
        {
            "topic_id"    : s["topic_id"],
            "start_msg_id": s["start_msg_id"],
            "end_msg_id"  : s["end_msg_id"],
            "msg_count"   : len(s["messages"]),
            "summary"     : s["summary"],
        }
        for s in segments
    ]


# ── Master build ──────────────────────────────────────────────────────────────

def build_all_checkpoints(messages: list[dict], force: bool = False) -> dict:
    """
    Build BOTH topic and 100-msg checkpoints.
    Returns {"topic": [...], "msg100": [...]}.
    """
    print("\n══ Building Topic Checkpoints ══")
    topic_cks = build_topic_checkpoints(messages, force=force)

    print("\n══ Building 100-Message Checkpoints ══")
    msg100_cks = build_msg100_checkpoints(messages, force=force)

    print(
        f"\n[checkpointer] Done. "
        f"{len(topic_cks)} topic checkpoints, "
        f"{len(msg100_cks)} msg-100 checkpoints."
    )
    return {"topic": topic_cks, "msg100": msg100_cks}


if __name__ == "__main__":
    from core.parser import get_messages
    msgs = get_messages()
    result = build_all_checkpoints(msgs)
    print(f"\nSample topic checkpoint: {result['topic'][0]}")
    print(f"Sample msg100 checkpoint: {result['msg100'][0]}")
