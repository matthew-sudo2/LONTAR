"""Shared pipeline defaults. Override via environment variables where noted."""

from __future__ import annotations

import os
from pathlib import Path

# Cohere embedding model — must be identical for embed and query paths.
COHERE_EMBED_MODEL: str = os.getenv("COHERE_EMBED_MODEL", "embed-v4.0")
COHERE_EMBED_MODELS: tuple[str, ...] = ("embed-v4.0", "embed-multilingual-v3.0")

# Chroma vector store
LONTAR_COLLECTION: str = os.getenv("LONTAR_COLLECTION", "lontar_ingredient_research")
CHROMA_PATH: Path = Path(os.getenv("CHROMA_PATH", "data/chroma"))

# Citation / recency guardrails (record passes if it meets either threshold)
MIN_CITATIONS: int = int(os.getenv("LONTAR_MIN_CITATIONS", "5"))
RECENT_YEAR: int = int(os.getenv("LONTAR_RECENT_YEAR", "2024"))

# Retrieval and embedding batching
DEFAULT_TOP_K: int = int(os.getenv("LONTAR_TOP_K", "12"))
DEFAULT_BATCH_SIZE: int = int(os.getenv("LONTAR_BATCH_SIZE", "48"))

# Research focus
DEFAULT_FOCUS: str = "Health & Clinical Therapy"
FOCUS_OPTIONS: tuple[str, ...] = (
    "Health & Clinical Therapy",
    "Agriculture & Botany",
    "Broad/Generic (No Filter)",
)

# LLM synthesis
DEFAULT_LLM_PROVIDER: str = "gemini"
DEFAULT_GEMINI_MODEL: str = "gemini-2.0-flash"
DEFAULT_GROQ_MODEL: str = "llama-3.3-70b-versatile"
GROQ_MODELS: tuple[str, ...] = (
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
    "mixtral-8x7b-32768",
    "gemma2-9b-it",
)
SUMMARY_GROQ_MODELS: tuple[str, ...] = (
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
)
DEFAULT_SUMMARY_MODEL: str = DEFAULT_GROQ_MODEL
DEFAULT_EMBED_FIELD: str = "auto"
DEFAULT_GEMINI_RETRIES: int = 2
DEFAULT_GEMINI_BACKOFF: float = 2.5

# Default ingredient set for CLI / API examples
DEFAULT_INGREDIENTS: tuple[str, ...] = ("Macadamia", "Turmeric", "Pepperberry")

# Warn in the UI when total ingestion page fetches exceed this count
INGESTION_PAGE_WARNING_THRESHOLD: int = 15
