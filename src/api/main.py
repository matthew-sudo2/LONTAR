import os 
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from src.api.schemas import IngestRequest, FilterEmbedRequest, FilterEmbedResponse, SynthesizeRequest
from src.ingestion import ingest_ingredients, StudyRecord
from src.phase2 import filter_records, build_chunks, embed_and_upsert
from src.phase3 import retrieve_sources, build_prompt, generate_report, generate_report_groq, Report

load_dotenv()

app = FastAPI(
    title="LONTAR API Backend",
    description="Synchronous functional API routing for the literature pipeline engine",
    version="1.0.0"
)

# Enable CORS 
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health_check(): 
    return {
        "status": "healthy", "engine": "LONTAR function pipeline ready"
        }

@app.post("/api/v1/ingest", response_model=list[StudyRecord])
async def endpoint_ingest(payload: IngestRequest): 
    try: 
        results = await ingest_ingredients (
            ingredients=payload.ingredients,
            openalex_pages=payload.openalex_pages,
            semantic_scholar_pages=payload.semantic_pages,
            pubmed_pages=payload.pubmed_pages,
            core_pages=payload.core_pages,
            per_page=payload.per_page,
            focus=payload.focus
        )
        return results 
    except Exception as e: 
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")

@app.post("/api/v1/filter-embed", response_model=FilterEmbedResponse)
def endpoint_filter_embed(payload: FilterEmbedRequest): 
    try: 
        raw_dicts = [record.model_dump() for record in payload.records]

        # Execute filtering
        filtered = filter_records(
            raw_dicts,
            min_citations=payload.min_citations,
            recent_year=payload.recent_year,
            ingredients=payload.ingredients,
            focus=payload.focus
        )
        
        # Build chunks and push to Chroma
        chunks = build_chunks(filtered)
        embed_and_upsert(
            chunks=chunks,
            chroma_path=payload.chroma_path,
            collection_name=payload.collection,
            cohere_model=payload.cohere_model,
            batch_size=payload.batch_size
        )
        return FilterEmbedResponse(
            success=True,
            input_count=len(raw_dicts),
            filtered_count=len(filtered),
            message="Successfully processed guardrails and updated local vector collection store."
        )
    except Exception as e: 
        raise HTTPException(status_code=500, detail=f"Filtering/Embedding phase failure: {str(e)}")
    
@app.post("/api/v1/synthesize", response_model=Report)
def endpoint_synthesize(payload: SynthesizeRequest): 
    try: 
        chroma_path = Path(getattr(payload, "chroma_path", Path("data") / "chroma"))
        collection_name = getattr(payload, "collection", "lontar_ingredient_research_v4")
        cohere_model = getattr(payload, "cohere_model", "embed-v4.0")
        sources = retrieve_sources(
            ingredients=payload.ingredients,
            chroma_path=chroma_path,
            collection_name=collection_name,
            cohere_model=cohere_model,
            top_k=payload.top_k,
            focus=payload.focus
        )
        if not sources: 
            raise HTTPException(
                status_code=404, 
                detail="No source reference literature found in database matching query items."
            )
        
        prompt = build_prompt(payload.ingredients, sources)
        report = generate_report_groq(
            prompt=prompt,
            model_name="llama-3.3-70b-versatile"
        )
        return report
    except HTTPException as http_exc: 
        raise http_exc
    except Exception as e: 
        raise HTTPException(status_code=500, detail=f"Synthesis Generation phase failure: {str(e)}")