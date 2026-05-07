"""
core/persona_extractor.py
=========================
Part 2 — Extract structured user persona from conversation data.

Strategy:
    1. Sample representative messages from across the full dataset.
    2. Split into batches of MAX_BATCH_MSGS messages.
    3. Extract partial personas per batch via Groq LLM.
    4. Merge all partial personas into one final JSON.
    5. Persist to PERSONA_JSON.
"""

from __future__ import annotations
import json
import random
from pathlib import Path
from groq import Groq

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    GROQ_API_KEY, LLM_MODEL, LLM_MAX_TOKENS,
    PERSONA_EXTRACTION_PROMPT, PERSONA_JSON,
)


# ── Constants ─────────────────────────────────────────────────────────────────

MAX_BATCH_MSGS = 150
SAMPLE_TOTAL   = 1500
RANDOM_SEED    = 42


# ── LLM call ─────────────────────────────────────────────────────────────────

def _extract_partial_persona(messages: list[dict]) -> dict:
    if not GROQ_API_KEY:
        raise EnvironmentError("GROQ_API_KEY not set.")

    text_block = "\n".join(
        f"[{m['speaker']}]: {m['text']}" for m in messages
    )
    prompt = PERSONA_EXTRACTION_PROMPT.format(messages=text_block)

    try:
        client = Groq(api_key=GROQ_API_KEY)
        response = client.chat.completions.create(
            model=LLM_MODEL,
            max_tokens=LLM_MAX_TOKENS,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = response.choices[0].message.content.strip()
        # Strip markdown fences if present
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        return json.loads(raw.strip())
    except Exception as exc:
        print(f"[persona_extractor] Batch failed: {exc}")
        return {}


# ── Merge partial personas ────────────────────────────────────────────────────

def _merge_personas(partials: list[dict]) -> dict:
    merged: dict = {
        "habits"              : [],
        "personal_facts"      : [],
        "personality_traits"  : [],
        "communication_style" : {
            "message_length"  : [],
            "tone"            : [],
            "emoji_usage"     : [],
            "notable_patterns": [],
        },
    }

    for p in partials:
        if not p:
            continue
        for key in ("habits", "personal_facts", "personality_traits"):
            merged[key].extend(p.get(key, []))
        cs = p.get("communication_style", {})
        for sub in ("message_length", "tone", "emoji_usage"):
            val = cs.get(sub)
            if val:
                merged["communication_style"][sub].append(val)
        merged["communication_style"]["notable_patterns"].extend(
            cs.get("notable_patterns", [])
        )

    def dedup(lst: list[str]) -> list[str]:
        seen = set()
        out  = []
        for item in lst:
            key = item.strip().lower()
            if key and key not in seen:
                seen.add(key)
                out.append(item.strip())
        return out

    for key in ("habits", "personal_facts", "personality_traits"):
        merged[key] = dedup(merged[key])

    def mode(lst: list[str]) -> str:
        if not lst:
            return "unknown"
        return max(set(lst), key=lst.count)

    cs = merged["communication_style"]
    cs["message_length"]   = mode(cs["message_length"])
    cs["tone"]             = mode(cs["tone"])
    cs["emoji_usage"]      = mode(cs["emoji_usage"])
    cs["notable_patterns"] = dedup(cs["notable_patterns"])

    return merged


# ── Public API ────────────────────────────────────────────────────────────────

def extract_persona(
    messages: list[dict],
    force: bool = False,
    path: Path = PERSONA_JSON,
) -> dict:
    if not force and path.exists():
        with open(path, encoding="utf-8") as fh:
            persona = json.load(fh)
        print(f"[persona_extractor] Loaded persona from cache ({path}).")
        return persona

    random.seed(RANDOM_SEED)
    sample = random.sample(messages, min(SAMPLE_TOTAL, len(messages)))
    sample.sort(key=lambda m: m["msg_id"])
    print(f"[persona_extractor] Sampled {len(sample)} messages for persona extraction.")

    partials = []
    for start in range(0, len(sample), MAX_BATCH_MSGS):
        batch = sample[start : start + MAX_BATCH_MSGS]
        batch_num = start // MAX_BATCH_MSGS + 1
        total_batches = (len(sample) + MAX_BATCH_MSGS - 1) // MAX_BATCH_MSGS
        print(f"[persona_extractor] Batch {batch_num}/{total_batches}…")
        partial = _extract_partial_persona(batch)
        partials.append(partial)

    persona = _merge_personas(partials)

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(persona, fh, indent=2, ensure_ascii=False)
    print(f"[persona_extractor] Persona saved → {path}")

    return persona


def load_persona(path: Path = PERSONA_JSON) -> dict:
    if not path.exists():
        raise FileNotFoundError("Persona not found. Run build_pipeline.py first.")
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


if __name__ == "__main__":
    from core.parser import get_messages
    msgs = get_messages()
    persona = extract_persona(msgs)
    print(json.dumps(persona, indent=2))