"""
core/parser.py
==============
Parses conversations.csv → flat list of Message objects in chronological order.

Each CSV row = one conversation (multi-line string).
Each line inside a row = one message ("User 1: ..." or "User 2: ...").

Output schema (each message dict):
{
    "msg_id"       : int,          # global sequential id (1-based, chronological)
    "conv_id"      : int,          # which CSV row (1-based)
    "turn_in_conv" : int,          # position inside that conversation (1-based)
    "speaker"      : "User 1" | "User 2",
    "text"         : str,          # raw utterance text
}
"""

import csv
import json
import re
from pathlib import Path
from tqdm import tqdm

import sys, os
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import CONVERSATIONS_CSV, MESSAGES_JSONL


# ── helpers ──────────────────────────────────────────────────────────────────

_MSG_RE = re.compile(r'^(User [12]):\s*(.*)', re.DOTALL)


def _parse_conversation(raw_text: str, conv_id: int) -> list[dict]:
    """
    Split a raw multi-line conversation string into individual message dicts.
    Lines that don't match 'User N: …' pattern are appended to previous message.
    """
    messages = []
    current_speaker = None
    current_text_parts = []

    for line in raw_text.splitlines():
        line = line.strip()
        if not line:
            continue
        m = _MSG_RE.match(line)
        if m:
            # flush previous
            if current_speaker is not None:
                messages.append({
                    "conv_id"      : conv_id,
                    "turn_in_conv" : len(messages) + 1,
                    "speaker"      : current_speaker,
                    "text"         : " ".join(current_text_parts).strip(),
                })
            current_speaker = m.group(1)
            current_text_parts = [m.group(2).strip()]
        else:
            # continuation of previous message
            if current_speaker is not None:
                current_text_parts.append(line)
            # else: orphan line before any speaker — ignore

    # flush last
    if current_speaker is not None and current_text_parts:
        messages.append({
            "conv_id"      : conv_id,
            "turn_in_conv" : len(messages) + 1,
            "speaker"      : current_speaker,
            "text"         : " ".join(current_text_parts).strip(),
        })

    return messages


# ── public API ───────────────────────────────────────────────────────────────

def parse_csv(csv_path: Path = CONVERSATIONS_CSV) -> list[dict]:
    """
    Read conversations.csv and return a flat list of message dicts,
    sorted chronologically (by conv_id, then turn_in_conv).
    Also assigns a global `msg_id`.
    """
    all_messages: list[dict] = []
    conv_id = 0

    with open(csv_path, newline="", encoding="utf-8", errors="replace") as fh:
        reader = csv.reader(fh)
        for row in tqdm(reader, desc="Parsing CSV rows"):
            if not row:
                continue
            raw_text = row[0]  # entire conversation in first column
            if not raw_text.strip():
                continue
            conv_id += 1
            msgs = _parse_conversation(raw_text, conv_id)
            all_messages.extend(msgs)

    # assign global sequential ids
    for idx, msg in enumerate(all_messages, start=1):
        msg["msg_id"] = idx

    print(f"[parser] Parsed {conv_id} conversations → {len(all_messages)} messages total")
    return all_messages


def save_messages(messages: list[dict], path: Path = MESSAGES_JSONL) -> None:
    """Persist parsed messages to JSONL for reuse."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for msg in messages:
            fh.write(json.dumps(msg, ensure_ascii=False) + "\n")
    print(f"[parser] Saved {len(messages)} messages → {path}")


def load_messages(path: Path = MESSAGES_JSONL) -> list[dict]:
    """Load persisted messages from JSONL."""
    messages = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                messages.append(json.loads(line))
    print(f"[parser] Loaded {len(messages)} messages from {path}")
    return messages


def get_messages(force_reparse: bool = False) -> list[dict]:
    """
    Return parsed messages. Uses cached JSONL if available.
    Set force_reparse=True to re-read the CSV.
    """
    if not force_reparse and MESSAGES_JSONL.exists():
        return load_messages()
    messages = parse_csv()
    save_messages(messages)
    return messages


if __name__ == "__main__":
    msgs = get_messages(force_reparse=True)
    print(f"Sample msg: {msgs[0]}")
    print(f"Sample msg: {msgs[100]}")
