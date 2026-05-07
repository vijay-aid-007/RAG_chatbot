"""
build_pipeline.py
=================
Master build script — run this ONCE to set up the entire system.

Steps executed in order:
    1. Parse CSV → flat messages (cached to storage/messages.jsonl)
    2. Build FAISS vector index (cached to storage/faiss.*)
    3. Build topic checkpoints + summaries (cached to storage/topic_checkpoints.json)
    4. Build 100-msg checkpoints (cached to storage/msg100_checkpoints.json)
    5. Extract user persona (cached to storage/persona.json)

Re-running is safe — each step checks its cache and skips if up-to-date.
Pass --force to rebuild everything from scratch.

Usage:
    python build_pipeline.py
    python build_pipeline.py --force
"""

import argparse
import time
from pathlib import Path

# ── CLI ───────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(description="Build the RAG pipeline")
parser.add_argument(
    "--force", action="store_true",
    help="Force rebuild all artifacts even if cache exists"
)
parser.add_argument(
    "--skip-persona", action="store_true",
    help="Skip persona extraction (saves API credits during testing)"
)
parser.add_argument(
    "--skip-summaries", action="store_true",
    help="Skip LLM summarisation — checkpoint summaries will be None"
)
args = parser.parse_args()

# ── Imports (after path is set) ───────────────────────────────────────────────

import sys
sys.path.insert(0, str(Path(__file__).parent))

from core.parser import get_messages
from core.vector_store import build_index
from core.checkpointer import build_all_checkpoints
from core.persona_extractor import extract_persona


def _banner(title: str) -> None:
    print(f"\n{'━'*60}")
    print(f"  {title}")
    print(f"{'━'*60}")


def main() -> None:
    total_start = time.time()

    # ── Step 1: Parse ─────────────────────────────────────────────────────────
    _banner("Step 1 / 4 — Parse conversations")
    t = time.time()
    messages = get_messages(force_reparse=args.force)
    print(f"  ✓  {len(messages):,} messages  [{time.time()-t:.1f}s]")

    # ── Step 2: Vector index ──────────────────────────────────────────────────
    _banner("Step 2 / 4 — Build FAISS vector index")
    t = time.time()
    build_index(messages, force=args.force)
    print(f"  ✓  Index built  [{time.time()-t:.1f}s]")

    # ── Step 3 & 4: Checkpoints ───────────────────────────────────────────────
    _banner("Step 3 / 4 — Build topic & 100-msg checkpoints")
    t = time.time()
    if args.skip_summaries:
        print("  ⚠  Skipping LLM summaries (--skip-summaries flag set).")
        print("     Checkpoints will have summary=None.")
        # Still run topic detection without LLM summarisation
        from core.topic_detector import (
            detect_topic_boundaries, build_topic_segments, save_topic_segments,
        )
        from config import MSG_CHECKPOINTS, TOPIC_CHECKPOINTS
        import json

        if args.force or not TOPIC_CHECKPOINTS.exists():
            bounds = detect_topic_boundaries(messages)
            segs   = build_topic_segments(messages, bounds)
            save_topic_segments(segs)

        if args.force or not MSG_CHECKPOINTS.exists():
            from config import MSG_CHECKPOINT_INTERVAL
            checkpoints = []
            for start in range(0, len(messages), MSG_CHECKPOINT_INTERVAL):
                w = messages[start : start + MSG_CHECKPOINT_INTERVAL]
                if w:
                    checkpoints.append({
                        "checkpoint_id": len(checkpoints) + 1,
                        "start_msg_id" : w[0]["msg_id"],
                        "end_msg_id"   : w[-1]["msg_id"],
                        "msg_count"    : len(w),
                        "summary"      : None,
                    })
            MSG_CHECKPOINTS.parent.mkdir(parents=True, exist_ok=True)
            with open(MSG_CHECKPOINTS, "w") as fh:
                json.dump(checkpoints, fh, indent=2)
            print(f"  ✓  {len(checkpoints)} msg-100 checkpoints (no summaries)")
    else:
        result = build_all_checkpoints(messages, force=args.force)
        print(
            f"  ✓  {len(result['topic'])} topic checkpoints, "
            f"{len(result['msg100'])} msg-100 checkpoints  [{time.time()-t:.1f}s]"
        )

    # ── Step 5: Persona ───────────────────────────────────────────────────────
    if not args.skip_persona:
        _banner("Step 4 / 4 — Extract user persona")
        t = time.time()
        persona = extract_persona(messages, force=args.force)
        n_habits  = len(persona.get("habits", []))
        n_facts   = len(persona.get("personal_facts", []))
        n_traits  = len(persona.get("personality_traits", []))
        print(
            f"  ✓  Persona extracted: "
            f"{n_habits} habits, {n_facts} facts, {n_traits} traits  "
            f"[{time.time()-t:.1f}s]"
        )
    else:
        print("\n  ⚠  Persona extraction skipped (--skip-persona flag).")

    # ── Done ──────────────────────────────────────────────────────────────────
    _banner(f"Pipeline complete — total {time.time()-total_start:.1f}s")
    print("  Run the chatbot with:  python app.py")
    print()


if __name__ == "__main__":
    main()
