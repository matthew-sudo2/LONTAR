# Project LONTAR 🌿
**Literature-Oriented Nutritional Tracking & Analysis RAG**

Project LONTAR is an automated data engineering and research synthesis pipeline designed to bridge the gap between complex nutritional science and food innovation. Built specifically for analyzing functional ingredient blends—like the **Tucker Dust Aussie Bush Spice Seasoning**—it automates the ingestion, conceptual alignment, and synthesis of global research papers into actionable health claim matrices.

##  The Mission
Manually searching through Google Scholar or PubMed for multiple ingredients is slow and prone to human error. LONTAR uses a **Headless RAG (Retrieval-Augmented Generation)** architecture to bypass weeks of manual literature reviews, delivering an evidence-based research roadmap in minutes.

##  The Tech Stack
Built for maximum efficiency using free-tier and open-source tools:
* **LLM / Synthesis:** [Google Gemini 2.0 Flash](https://aistudio.google.com/) (CLI/API default) or [Groq](https://groq.com/) (Streamlit UI default)
* **Ingestion APIs:** [OpenAlex](https://openalex.org/), [Semantic Scholar](https://www.semanticscholar.org/product/api), PubMed, and optionally [CORE](https://core.ac.uk/)
* **Embeddings:** [Cohere Embed v4.0](https://cohere.com/embeddings) (canonical default; override with `COHERE_EMBED_MODEL`)
* **Vector Store:** [ChromaDB](https://www.trychroma.com/) (local/embedded)
* **Orchestration:** Python (Asyncio, Pydantic, FastAPI, Streamlit)

##  Core Architecture
LONTAR operates as a high-density "Compression Funnel":
1.  **Aggressive Fetching:** Concurrent asynchronous requests batch-query global databases.
2.  **Keyword Relevance Triage:** `relevance.py` scores abstracts against ingredient aliases and focus-profile term lists (deterministic heuristic, not an ML classifier).
3.  **AI Benefit Summarization (optional):** `summarize.py` runs each guardrail-passing record through Groq to extract direction (benefit/risk/neutral/unrelated), mechanism, and confidence. When enabled, `benefit_summary` is embedded instead of the raw abstract.
4.  **Deterministic Guardrails:** Citation-count and recency thresholds (default: ≥5 citations **or** published ≥2024) prune low-signal records before embedding.
5.  **Vector Retrieval + Re-filter:** Chroma nearest-neighbor search, then relevance re-scoring on retrieved chunks.
6.  **Context Stuffing:** An LLM synthesizes filtered abstracts into a structured health-claim report, preferring benefit-direction sources for claims and risk-direction sources for safety.

**Not implemented:** peer-review or venue-type verification. Claims about "legally grounded" output should be treated as aspirational until that layer exists.

## 📂 Repository Structure
```text
├── app_frontend.py       # Streamlit UI (3-tab pipeline)
├── src/
│   ├── config.py         # Shared defaults (embed model, collection, guardrails)
│   ├── ingestion.py      # Async concurrent API fetching (Phase 1)
│   ├── relevance.py      # Ingredient aliases + keyword relevance scoring
│   ├── summarize.py      # Groq health-benefit summarization per record
│   ├── phase2.py         # Guardrail filter + summarize + Cohere embed + Chroma upsert
│   ├── phase3.py         # Vector retrieval + LLM report synthesis
│   └── api/
│       ├── main.py       # FastAPI REST endpoints
│       └── schemas.py    # Request/response models
├── data/                 # Local JSON dumps and ChromaDB storage
├── outputs/              # Generated Markdown/JSON reports
├── tests/
│   └── test_relevance.py
└── requirements.txt
```

## Setup (Windows PowerShell)
1. Create and activate the virtual environment:
	```powershell
	python -m venv .venv
	.\.venv\Scripts\Activate.ps1
	```
2. Upgrade pip and install dependencies:
	```powershell
	python -m pip install --upgrade pip
	python -m pip install -r requirements.txt
	```
3. Create a local `.env` file and paste API keys (do not commit it).

## Environment Variables
Fill in these values in your local `.env` file:
- `GEMINI_API_KEY`
- `SEMANTIC_SCHOLAR_API_KEY`
- `COHERE_API_KEY`
- `GROQ_API_KEY`
- `CORE_API_KEY` (optional, enables CORE ingestion)
- `OPENALEX_EMAIL` (recommended for OpenAlex polite pool)

Optional pipeline overrides (see `src/config.py`):
- `COHERE_EMBED_MODEL` — default `embed-v4.0`
- `LONTAR_COLLECTION` — default `lontar_ingredient_research`
- `LONTAR_MIN_CITATIONS` — default `5`
- `LONTAR_RECENT_YEAR` — default `2024`

## Quick Verification
```powershell
python -c "import google.genai, openalex, semanticscholar, cohere, chromadb, pandas, pydantic, dotenv; print('OK')"
```

## Phase 1: Ingestion Run
Run the ingestion script with a list of ingredients. Results are written to `data/ingestion.json` by default.
```powershell
python -m src.ingestion Macadamia Turmeric Pepperberry
```
Optional flags:
- `--openalex-pages 2`
- `--semantic-pages 2`
- `--pubmed-pages 2`
- `--core-pages 2`
- `--per-page 25`
- `--out data/ingestion.csv`

CORE ingestion is enabled when `CORE_API_KEY` is set in your environment.

## Phase 2: Guardrails + Chroma Embedding
Phase 2 filters low-quality records, deduplicates, and embeds abstracts into ChromaDB.
```powershell
python -m src.phase2
```
Optional summarization flags:
- `--summarize` — run Groq health-benefit summarization before embedding
- `--summary-model llama-3.3-70b-versatile`
- `--drop-risk-and-unrelated` — exclude risk/unrelated records from Chroma
- `--embed-field auto` — embed benefit_summary when available (default with `--summarize`)

Defaults (shared with Streamlit/API via `src/config.py`):
- `--min-citations 5`
- `--recent-year 2024`
- `--collection lontar_ingredient_research`
- `--chroma-path data/chroma`
- `--cohere-model embed-v4.0`

## Phase 3: Generation Engine
Phase 3 retrieves the best matches from ChromaDB and generates the final report.
```powershell
python -m src.phase3 Macadamia Turmeric Pepperberry
```
Key options:
- `--top-k 12`
- `--focus "Health & Clinical Therapy"`
- `--collection lontar_ingredient_research`
- `--chroma-path data/chroma`
- `--cohere-model embed-v4.0`
- `--gemini-model gemini-2.0-flash`
- `--llm-provider groq`
- `--groq-model llama-3.3-70b-versatile`
- `--out-json outputs/report.json`
- `--out-md outputs/report.md`

## Streamlit UI
```powershell
streamlit run app_frontend.py
```

## FastAPI
```powershell
uvicorn src.api.main:app --reload
```

## One-Command Pipeline (Phase 1-3)
Run everything end-to-end with the default ingredient list:
```powershell
python -m src.pipeline
```
Or pass your own ingredients:
```powershell
python -m src.pipeline "macadamia nuts" "hemp seeds" "nutritional yeast" "yellow mustard powder" "Himalayan salt" "turmeric" "ginger" "cinnamon" "pepperberry"
```

## Relevance Filter Notes
The keyword filter in `relevance.py` is a fast pre-filter tuned for obvious noise (e.g., wastewater/materials papers). When `--summarize` is enabled, `summarize.py` adds a second LLM pass that classifies effect direction per record. "Broad/Generic (No Filter)" applies no focus-profile scoring.
