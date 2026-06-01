# Project LONTAR 🌿
**Literature-Oriented Nutritional Tracking & Analysis RAG**

Project LONTAR is an automated data engineering and research synthesis pipeline designed to bridge the gap between complex nutritional science and food innovation. Built specifically for analyzing functional ingredient blends—like the **Tucker Dust Aussie Bush Spice Seasoning**—it automates the ingestion, conceptual alignment, and synthesis of global research papers into actionable health claim matrices.

##  The Mission
Manually searching through Google Scholar or PubMed for multiple ingredients is slow and prone to human error. LONTAR uses a **Headless RAG (Retrieval-Augmented Generation)** architecture to bypass weeks of manual literature reviews, delivering a legally grounded "Evidence-Based Roadmap" in minutes.

##  The Tech Stack
Built for maximum efficiency using free-tier and open-source tools:
* **LLM / Synthesis:** [Google Gemini 2.0 Flash](https://aistudio.google.com/) (1M token context window)
* **Ingestion APIs:** [OpenAlex API](https://openalex.org/) & [Semantic Scholar Graph API](https://www.semanticscholar.org/product/api)
* **Embeddings:** [Cohere Multilingual Embed v4.0](https://cohere.com/embeddings)
* **Vector Store:** [ChromaDB](https://www.trychroma.com/) (Local/Embedded)
* **Orchestration:** Python (Asyncio, Pandas, Pydantic)

##  Core Architecture
LONTAR operates as a high-density "Compression Funnel":
1.  **Aggressive Fetching:** Concurrent asynchronous requests batch-query global databases.
2.  **Multilingual Alignment:** Maps non-English abstracts (e.g., German, Spanish, Mandarin) into a unified conceptual vector space.
3.  **Deterministic Guardrails:** Algorithmic pruning drops studies with low citation counts or unverified peer-review status before they hit the LLM.
4.  **Context Stuffing:** Leverages Gemini’s massive context window to synthesize hundreds of filtered abstracts in a single, token-efficient pass.

## 📂 Repository Structure
```text
├── src/
│   ├── ingestion.py      # Async concurrent API fetching
│   ├── guardrail.py      # Rule-based metadata pruning & validation
│   ├── vector_engine.py  # ChromaDB & Cohere embedding management
│   └── synthesizer.py    # Gemini 2.0 Flash report generation
├── data/                 # Local JSON dumps and ChromaDB storage
├── outputs/              # Generated Markdown/PDF reports
└── requirements.txt      # Project dependencies
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
3. Create your local env file and paste API keys:
	```powershell
	Copy-Item .env.example .env
	```

## Environment Variables
Fill in these values in your local `.env` file (do not commit it):
- `GEMINI_API_KEY`
- `SEMANTIC_SCHOLAR_API_KEY`
- `COHERE_API_KEY`
- `GROQ_API_KEY`

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
python -m src.phase2 --min-citations 5 --recent-year 2024
```
Key options:
- `--input data/ingestion.json`
- `--filtered-out data/filtered.json`
- `--collection lontar_ingredient_research`
- `--chroma-path data/chroma`
- `--cohere-model embed-multilingual-v3.0`

## Phase 3: Generation Engine
Phase 3 retrieves the best matches from ChromaDB and generates the final report with Gemini.
```powershell
python -m src.phase3 Macadamia Turmeric Pepperberry
```
Key options:
- `--top-k 12`
- `--collection lontar_ingredient_research`
- `--chroma-path data/chroma`
- `--cohere-model embed-multilingual-v3.0`
- `--gemini-model gemini-2.0-flash`
- `--gemini-retries 2`
- `--gemini-backoff 2.5`
- `--llm-provider groq`
- `--groq-model llama-3.3-70b-versatile`
- `--out-json outputs/report.json`
- `--out-md outputs/report.md`
