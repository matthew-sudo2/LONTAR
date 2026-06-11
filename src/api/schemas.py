from pydantic import BaseModel, Field 
from typing import List, Optional
from src.ingestion import StudyRecord
from src.phase3 import Report

class IngestRequest(BaseModel): 
    ingredients: List[str] = Field(default=["Macadamia", "Turmeric", "Pepperberry"])
    openalex_pages: int = 2 
    semantic_pages: int = 2 
    pubmed_pages: int = 2 
    core_pages: int = 2 
    per_page: int = 2 

class FilterEmbedRequest(BaseModel): 
    records: list[StudyRecord]
    min_citations: int = Field(default=0, description="Minimum citation count threshold")
    recent_year: int = Field(default=2018, description="Filter for articles published from this year onwards")
        
    chroma_path: str = Field(default="data/chroma")
    collection: str = Field(default="lontar_ingredient_research_v4")
    cohere_model: str = Field(default="embed-v4.0")
    batch_size: int = Field(default=64)

class FilterEmbedResponse(BaseModel): 
    success: bool 
    input_count: int 
    filtered_count: int 
    message: str 

class SynthesizeRequest(BaseModel): 
    ingredients: List[str]
    chroma_path: str = "data/chroma"
    collection: str = "lontar_ingredient_research"
    cohere_model: str = "embed-multilingual-v3.0"
    top_k: int = 15
    llm_provider: str = Field(default="gemini", description="Options: gemini, groq")
    gemini_model: str = "gemini-2.0-flash"
    groq_model: str = "llama-3.3-70b-versatile"
    gemini_retries: int = 3
    gemini_backoff: int = 2