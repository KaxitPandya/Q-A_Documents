"""
Document Q&A Assistant — Configuration & Constants
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ── OpenAI ──────────────────────────────────────────────────
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")

# Modern model defaults (updated 2025+)
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSIONS = 1536  # default for text-embedding-3-small

# Model options exposed in the UI
CHAT_MODELS = [
    "gpt-4o-mini",
    "gpt-4o",
    "gpt-4-turbo",
    "gpt-3.5-turbo",
]

# ── Chunking strategies ────────────────────────────────────
CHUNKING_STRATEGIES = ["Recursive (fast)", "Semantic (smart)", "Parent-Child (precise)"]

# ── Pipeline modes (replaces old presets) ──────────────────
PIPELINE_MODES = [
    "🧠 Adaptive (recommended)",
    "🤖 Agent",
    "🔧 Manual",
]

# ── Conversation summarization ─────────────────────────────
SUMMARIZATION_THRESHOLD = 8   # summarize when history exceeds this many messages
SUMMARY_KEEP_RECENT = 4       # always keep this many recent messages verbatim

# ── ChromaDB ────────────────────────────────────────────────
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")

# ── Token costs (USD per 1K tokens, as of 2025) ────────────
TOKEN_COSTS = {
    "gpt-4o-mini":   {"input": 0.00015,  "output": 0.0006},
    "gpt-4o":        {"input": 0.0025,   "output": 0.01},
    "gpt-4-turbo":   {"input": 0.01,     "output": 0.03},
    "gpt-3.5-turbo": {"input": 0.0005,   "output": 0.0015},
    EMBEDDING_MODEL: {"input": 0.00002,  "output": 0.0},
}

# ── Retry / resilience ─────────────────────────────────────
MAX_RETRIES = 3
RETRY_BACKOFF = 2          # exponential back-off base (seconds)
REQUEST_TIMEOUT = 60       # seconds

# ── Logging ─────────────────────────────────────────────────
LOG_DIR = os.getenv("LOG_DIR", "./logs")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# ── Supported file types ───────────────────────────────────
SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md", ".csv"}
