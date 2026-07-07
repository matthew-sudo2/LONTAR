from pydantic import BaseModel, Field
from typing import List, Optional

from src.config import (
    CHROMA_PATH,
    COHERE_EMBED_MODEL,
    DEFAULT_BATCH_SIZE,
    DEFAULT_EMBED_FIELD,
    DEFAULT_FOCUS,
    DEFAULT_GEMINI_BACKOFF,
    DEFAULT_GEMINI_MODEL,
    DEFAULT_GEMINI_RETRIES,
    DEFAULT_GROQ_MODEL,
    DEFAULT_INGREDIENTS,
    DEFAULT_LLM_PROVIDER,
    DEFAULT_SUMMARY_MODEL,
    DEFAULT_TOP_K,
    LONTAR_COLLECTION,
    MIN_CITATIONS,
    RECENT_YEAR,
)
from src.ingestion import StudyRecord
from src.phase3 import Report


class IngestRequest(BaseModel):
    ingredients: List[str] = Field(default=list(DEFAULT_INGREDIENTS))
    openalex_pages: int = 2
    semantic_pages: int = 2
    pubmed_pages: int = 2
    core_pages: int = 2
    per_page: int = 2
    focus: Optional[str] = DEFAULT_FOCUS


class FilterEmbedRequest(BaseModel):
    records: list[StudyRecord]
    min_citations: int = Field(
        default=MIN_CITATIONS,
        description="Minimum citation count threshold",
    )
    recent_year: int = Field(
        default=RECENT_YEAR,
        description="Filter for articles published from this year onwards",
    )
    ingredients: Optional[List[str]] = None
    focus: Optional[str] = None

    chroma_path: str = Field(default=str(CHROMA_PATH))
    collection: str = Field(default=LONTAR_COLLECTION)
    cohere_model: str = Field(default=COHERE_EMBED_MODEL)
    batch_size: int = Field(default=DEFAULT_BATCH_SIZE)
    summarize: bool = Field(
        default=True,
        description="Run AI health-benefit summarization before embedding.",
    )
    summary_model: str = Field(default=DEFAULT_SUMMARY_MODEL)
    drop_risk_and_unrelated: bool = Field(default=False)
    embed_field: str = Field(default=DEFAULT_EMBED_FIELD)


class FilterEmbedResponse(BaseModel):
    success: bool
    input_count: int
    filtered_count: int
    message: str


class SynthesizeRequest(BaseModel):
    ingredients: List[str]
    chroma_path: str = str(CHROMA_PATH)
    collection: str = LONTAR_COLLECTION
    cohere_model: str = COHERE_EMBED_MODEL
    top_k: int = DEFAULT_TOP_K
    focus: Optional[str] = DEFAULT_FOCUS
    llm_provider: str = Field(default=DEFAULT_LLM_PROVIDER, description="Options: gemini, groq")
    gemini_model: str = DEFAULT_GEMINI_MODEL
    groq_model: str = DEFAULT_GROQ_MODEL
    gemini_retries: int = DEFAULT_GEMINI_RETRIES
    gemini_backoff: int = int(DEFAULT_GEMINI_BACKOFF)
