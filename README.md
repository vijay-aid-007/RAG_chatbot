# RAG Conversation Chatbot

A production-ready Retrieval-Augmented Generation (RAG) system that processes a large conversation dataset, detects topic changes chronologically, extracts a structured user persona, and powers an intelligent chatbot.

---

## Live Demo
🔗 **Chatbot URL:** https://huggingface.co/spaces/vijay036/rag_chatbot 
🎥 **Video Demo:** [ADD YOUR LOOM URL HERE]  
💻 **GitHub:** https://github.com/vijay-aid-007/RAG_chatbot

---

## Table of Contents
1. [Architecture Overview](#architecture-overview)
2. [How Topic Changes Are Detected](#how-topic-changes-are-detected)
3. [How Retrieval Works](#how-retrieval-works)
4. [How Persona Is Built](#how-persona-is-built)
5. [Project Structure](#project-structure)
6. [Setup & Installation](#setup--installation)
7. [Running the System](#running-the-system)
8. [Cloud Deployment](#cloud-deployment)
9. [Configuration Reference](#configuration-reference)

---

## Architecture Overview

```
conversations.csv
      │
      ▼
  core/parser.py          → flat chronological message list
      │
      ├──► core/topic_detector.py   → topic segments + boundaries
      │         │
      │         ▼
      │    core/checkpointer.py     → topic summaries (Groq LLM)
      │                             → 100-msg summaries (Groq LLM)
      │
      ├──► core/vector_store.py     → FAISS index of message chunks
      │
      └──► core/persona_extractor.py → structured JSON persona (Groq LLM)

At query time:
  user question
      │
      ▼
  core/retriever.py        → topic summaries (cosine) + chunks (FAISS)
      │
      ▼
  core/chatbot.py          → Groq LLM (llama-3.3-70b-versatile) → answer
      │
      ▼
  app.py (Gradio UI)
```

---

## How Topic Changes Are Detected

**Method: Sliding Window Cosine Similarity**

Messages are processed strictly in chronological order (by `msg_id`). Topic boundaries are identified as follows:

1. **Windowing** — A sliding window of `W = 5` consecutive messages is formed at each position. Each window is concatenated into a single text block.

2. **Embedding** — Every window is encoded into a 384-dimensional dense vector using `all-MiniLM-L6-v2` (a lightweight sentence transformer, no GPU required).

3. **Similarity scoring** — The cosine similarity between adjacent window embeddings `sim(w_i, w_{i+1})` is computed.

4. **Boundary detection** — A topic boundary is declared at position `i+1` when:
   - `sim(w_i, w_{i+1}) < 0.35` (configurable threshold), **AND**
   - At least 5 messages have elapsed since the last boundary (prevents micro-splits).

5. **Output** — Each segment between boundaries forms one `TopicSegment`, which is then summarised via the Groq LLM.

**Why this works:**  
Single-message comparison is noisy (a single off-topic utterance would fire false boundaries). The window smooths over noise while still catching genuine topic drift. The threshold of 0.35 is set conservatively to prefer under-splitting, so each checkpoint represents a meaningful, coherent topic.

All parameters (`TOPIC_WINDOW_SIZE`, `TOPIC_SIMILARITY_THRESH`, `TOPIC_MIN_SEGMENT_LEN`) are tunable in `config.py`.

---

## How Retrieval Works

Retrieval is two-stage and runs at query time:

### Stage 1 — Topic Summary Retrieval
- All topic summaries are embedded into vectors once (lazily cached in RAM).
- The query is embedded and cosine similarity is computed against every topic summary vector.
- The top `TOP_K_TOPICS = 3` most relevant summaries are returned.

### Stage 2 — Message Chunk Retrieval (FAISS)
- At index-build time, messages are grouped into non-overlapping chunks of `CHUNK_SIZE = 8` messages.
- Each chunk is embedded and stored in a **FAISS `IndexFlatIP`** (exact inner-product, equivalent to cosine similarity on L2-normalised vectors).
- At query time, the query vector is searched against the FAISS index; top `TOP_K_CHUNKS = 5` chunks are returned.

### Stage 3 — Context Assembly
Both results are formatted into a structured context string and passed to the Groq LLM alongside the persona:

```
=== RELEVANT TOPIC SUMMARIES ===
[Topic 3 | msgs 120–180 | sim=0.72]
Users discussed outdoor activities, hiking trails...

=== RELEVANT MESSAGE CHUNKS ===
[Chunk: msgs 125–132 | sim=0.68]
[User 1]: I hiked the Everglades last weekend...
```

**Why two stages?**  
Topic summaries provide high-level thematic coverage (good for broad questions). Raw chunks provide exact grounding (good for specific fact retrieval). Combining both gives the LLM both context and precision.

---

## How Persona Is Built

1. **Sampling** — Messages are sampled across the full dataset stratified chronologically.

2. **Batch extraction** — Messages are processed in batches. For each batch, the Groq LLM is prompted to return a structured JSON persona covering:
   - `habits` — behavioural patterns (sleep schedule, food, exercise)
   - `personal_facts` — concrete facts (jobs, pets, relationships, locations)
   - `personality_traits` — inferred character (humorous, empathetic, curious)
   - `communication_style` — message length, tone, emoji usage, notable patterns

3. **Merge** — Partial personas from all batches are merged:
   - List fields (`habits`, `personal_facts`, `personality_traits`) → **deduplicated union** (case-insensitive)
   - Categorical fields (`tone`, `message_length`, `emoji_usage`) → **mode vote** across all batches

4. **Grounding** — The extraction prompt explicitly instructs the LLM: *"Base every item on explicit signals in the messages. Do NOT guess or invent."*

5. **Persistence** — Final persona stored as `storage/persona.json`.

### Persona storage format (`persona.json`):
```json
{
  "habits": ["reading", "cooking", "gardening", "hiking"],
  "personal_facts": ["has two dogs named Jack and Max", "stay-at-home mom", "three kids"],
  "personality_traits": ["loves helping others", "sense of humor", "enthusiastic"],
  "communication_style": {
    "tone": "casual",
    "message_length": "medium",
    "emoji_usage": "frequent"
  }
}
```

---

## Project Structure

```
rag_chatbot/
├── config.py                   # All tunable parameters (single source of truth)
├── build_pipeline.py           # Master build script (run once)
├── app.py                      # Gradio chatbot UI
├── requirements.txt            # Python dependencies
├── .env.example                # Environment variable template
│
├── core/
│   ├── parser.py               # CSV → flat message list
│   ├── embedder.py             # SentenceTransformer wrapper
│   ├── topic_detector.py       # Sliding window topic detection
│   ├── checkpointer.py         # Topic + 100-msg checkpoint builder
│   ├── summariser.py           # Groq LLM summarisation
│   ├── vector_store.py         # FAISS index build + search
│   ├── retriever.py            # Two-stage RAG retrieval
│   ├── persona_extractor.py    # Structured persona extraction
│   └── chatbot.py              # RAG + persona answer generation
│
├── data/
│   └── conversations.csv       # Input dataset
│
└── storage/                    # Auto-created by build pipeline
    ├── faiss.index             # FAISS vector index
    ├── faiss_meta.jsonl        # Chunk metadata
    ├── topic_checkpoints.json  # Topic segments + summaries
    ├── msg100_checkpoints.json # 100-msg checkpoints
    └── persona.json            # Extracted user persona
```

---

## Setup & Installation

### Prerequisites
- Python 3.10+
- A Groq API key — free at [console.groq.com](https://console.groq.com)

### Steps

```bash
# 1. Clone the repo
git clone https://github.com/vijay-aid-007/RAG_chatbot
cd rag-chatbot

# 2. Create a virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Mac/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set up environment variables
# Create a .env file in the project root and add:
# GROQ_API_KEY=your_groq_api_key_here

# 5. Place the dataset
# Copy conversations.csv into the data/ folder
```

---

## Running the System

### Step 1 — Build the pipeline (run once)

```bash
python build_pipeline.py
```

| Step | Estimated Time |
|---|---|
| Parse CSV | ~30 seconds |
| Build FAISS index | ~5 minutes |
| Topic checkpoints (Groq LLM) | ~15–60 minutes |
| 100-msg checkpoints (Groq LLM) | ~10–30 minutes |
| Persona extraction (Groq LLM) | ~5 minutes |

### Step 2 — Launch the chatbot

```bash
python app.py
```

Opens at `http://localhost:7860`

### Example questions to ask

```
"What kind of person is this user?"
"What are their daily habits?"
"How do they communicate — tone, style, emoji usage?"
"What are their hobbies and interests?"
"What personal facts do you know about this person?"
"Which topics appear most often in the conversations?"
"Does this person have any pets?"
```

---

## Cloud Deployment

### Hugging Face Spaces (free, recommended)
1. Go to [huggingface.co](https://huggingface.co) → New Space → Gradio SDK
2. Upload all project files
3. Go to Space Settings → Secrets → add `GROQ_API_KEY`
4. Upload pre-built `storage/` folder (FAISS index + checkpoints + persona)

### Railway / Render
1. Connect your GitHub repo
2. Add environment variable: `GROQ_API_KEY=your_key`
3. Set start command: `python app.py`
4. Mount persistent volume at `/app/storage`

---

## Tech Stack

| Component | Technology |
|---|---|
| Embeddings | `sentence-transformers` (`all-MiniLM-L6-v2`) — local, no GPU needed |
| Vector store | FAISS (`IndexFlatIP`) |
| LLM | Groq API (`llama-3.3-70b-versatile`) |
| UI | Gradio 5.x |
| Language | Python 3.11 |

---

## Configuration Reference

All parameters are in `config.py`:

| Parameter | Default | Description |
|---|---|---|
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Sentence transformer model |
| `TOPIC_WINDOW_SIZE` | `5` | Messages per sliding window |
| `TOPIC_SIMILARITY_THRESH` | `0.35` | Cosine threshold for topic change |
| `TOPIC_MIN_SEGMENT_LEN` | `5` | Min messages before allowing a split |
| `MSG_CHECKPOINT_INTERVAL` | `100` | Messages per 100-msg checkpoint |
| `CHUNK_SIZE` | `8` | Messages per FAISS chunk |
| `TOP_K_CHUNKS` | `5` | Chunks returned per query |
| `TOP_K_TOPICS` | `3` | Topic summaries returned per query |
| `LLM_MODEL` | `llama-3.3-70b-versatile` | Groq model used for all LLM calls |
| `GROQ_API_KEY` | from `.env` | Groq API key |
