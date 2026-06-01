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
