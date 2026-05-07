"""
config.py — Central configuration for RAG Chatbot system.
All tunable parameters live here. Change values here only.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR        = Path(__file__).parent
DATA_DIR        = BASE_DIR / "data"
STORAGE_DIR     = BASE_DIR / "storage"          # persisted artifacts
STORAGE_DIR.mkdir(exist_ok=True)

CONVERSATIONS_CSV   = DATA_DIR / "conversations.csv"
MESSAGES_JSONL      = STORAGE_DIR / "messages.jsonl"          # flat parsed messages
TOPIC_CHECKPOINTS   = STORAGE_DIR / "topic_checkpoints.json"
MSG_CHECKPOINTS     = STORAGE_DIR / "msg100_checkpoints.json"
PERSONA_JSON        = STORAGE_DIR / "persona.json"
FAISS_INDEX_PATH    = STORAGE_DIR / "faiss.index"
FAISS_META_PATH     = STORAGE_DIR / "faiss_meta.jsonl"

# ── Embedding model ───────────────────────────────────────────────────────────
# Lightweight model that runs CPU-only, no GPU needed.
EMBEDDING_MODEL = "all-MiniLM-L6-v2"          # 80 MB, 384-dim, fast

# ── Topic detection ───────────────────────────────────────────────────────────
TOPIC_WINDOW_SIZE       = 5      # messages to form a rolling context window
TOPIC_SIMILARITY_THRESH = 0.35   # cosine drop below this → topic change
TOPIC_MIN_SEGMENT_LEN   = 5      # never split a segment shorter than this

# ── 100-message checkpoint ───────────────────────────────────────────────────
MSG_CHECKPOINT_INTERVAL = 100

# ── Retrieval ────────────────────────────────────────────────────────────────
TOP_K_CHUNKS    = 5    # raw message chunks to retrieve
TOP_K_TOPICS    = 3    # topic summaries to retrieve
CHUNK_SIZE      = 8    # messages per chunk stored in FAISS

# ── LLM (Groq) ───────────────────────────────────────────────────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
LLM_MODEL = "llama-3.3-70b-versatile"

LLM_MAX_TOKENS = 1024

# ── Summarisation prompt templates ───────────────────────────────────────────
TOPIC_SUMMARY_PROMPT = (
    "You are summarising one topic segment of a conversation.\n"
    "Be concise (3-5 sentences). Focus on: what was discussed, "
    "key facts shared, and emotional tone.\n\n"
    "Conversation segment:\n{segment}\n\nSummary:"
)

MSG100_SUMMARY_PROMPT = (
    "Summarise the following 100 messages from a conversation in 5-7 sentences.\n"
    "Capture the main themes, facts, and any notable moments.\n\n"
    "Messages:\n{segment}\n\nSummary:"
)

PERSONA_EXTRACTION_PROMPT = (
    "You are a persona analyst. Study the conversation messages below and extract a "
    "structured JSON persona. Return ONLY valid JSON — no prose, no markdown.\n\n"
    "Schema:\n"
    '{{\n'
    '  "habits": ["..."],\n'
    '  "personal_facts": ["..."],\n'
    '  "personality_traits": ["..."],\n'
    '  "communication_style": {{\n'
    '    "message_length": "short|medium|long",\n'
    '    "tone": "casual|formal|mixed",\n'
    '    "emoji_usage": "none|occasional|frequent",\n'
    '    "notable_patterns": ["..."]\n'
    "  }}\n"
    "}}\n\n"
    "Base every item on explicit signals in the messages. "
    "Do NOT guess or invent.\n\n"
    "Messages:\n{messages}\n\nJSON:"
)