import asyncio
import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from streamlit.errors import StreamlitSecretNotFoundError

from src.ingestion import ingest_ingredients, records_to_dicts
from src.phase2 import build_chunks, embed_and_upsert, filter_records
from src.phase3 import (
    build_prompt,
    generate_report,
    generate_report_groq,
    retrieve_sources,
)


SECRET_KEYS = [
    "GEMINI_API_KEY",
    "SEMANTIC_SCHOLAR_API_KEY",
    "COHERE_API_KEY",
    "GROQ_API_KEY",
    "CORE_API_KEY",
    "OPENALEX_EMAIL",
]

CHROMA_PATH = Path("data") / "chroma"


def hydrate_environment_from_streamlit_secrets() -> None:
    """Expose Streamlit Cloud secrets through os.environ for pipeline modules."""
    try:
        for key in SECRET_KEYS:
            if key in st.secrets and not os.getenv(key):
                os.environ[key] = str(st.secrets[key])
    except StreamlitSecretNotFoundError:
        pass


def clean_ingredients(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def render_chatbot_matrix(matrix_list: list[dict]) -> None:
    for entry in matrix_list:
        if not isinstance(entry, dict) or "ingredient" not in entry:
            continue

        ingredient = entry.get("ingredient", "Unknown Ingredient")
        claim = entry.get("claim", "No compiled health claim description.")
        dois = entry.get("evidence_dois", [])
        titles = entry.get("evidence_titles", [])

        with st.chat_message("assistant"):
            st.markdown(f"### Ingredient: {ingredient}")
            st.markdown(f"**Identified Health Claim:** `{claim}`")

            if titles:
                st.markdown("**Supporting Academic Evidence:**")
                for idx, title in enumerate(titles):
                    doi_code = dois[idx] if idx < len(dois) else None
                    if doi_code:
                        st.markdown(
                            f"{idx + 1}. *{title}* - "
                            f"[View Study (DOI: {doi_code})](https://doi.org/{doi_code})"
                        )
                    else:
                        st.markdown(f"{idx + 1}. *{title}*")
            elif dois:
                st.markdown("**Supporting Academic Evidence (DOIs):**")
                for idx, doi_code in enumerate(dois):
                    st.markdown(
                        f"{idx + 1}. [View Study Document (DOI: {doi_code})](https://doi.org/{doi_code})"
                    )
            else:
                st.markdown("*No explicit publication citations were bound to this claim.*")


def render_synergy_matrix(synergy_list: list[dict]) -> None:
    for entry in synergy_list:
        if not isinstance(entry, dict):
            st.write(entry)
            continue

        ingredients = entry.get("ingredients", [])
        formula_title = " + ".join(ingredients) if ingredients else "Unknown Blend"
        mechanism = entry.get("mechanism", "No mechanism description provided.")
        dois = entry.get("evidence_dois", [])

        with st.chat_message("assistant"):
            st.markdown(f"### Synergistic Combo: {formula_title}")
            st.markdown(f"**Biological Mechanism:** {mechanism}")

            if dois:
                st.markdown("**Evidence Links:**")
                for idx, doi in enumerate(dois):
                    st.markdown(
                        f"{idx + 1}. [Open Source Publication (DOI: {doi})](https://doi.org/{doi})"
                    )


def render_limitations(limitations_list: list[str]) -> None:
    with st.chat_message("assistant"):
        st.markdown("### Identified Research Gaps & Limitations")
        for idx, item in enumerate(limitations_list):
            st.markdown(f"* **Gap {idx + 1}:** {item}")


def render_report(report_data: dict | list | str) -> None:
    if isinstance(report_data, list):
        st.subheader("Analysis Results")
        render_chatbot_matrix(report_data)
        return

    if isinstance(report_data, str):
        st.markdown(report_data)
        return

    if not isinstance(report_data, dict):
        st.write(report_data)
        return

    extracted_text = (
        report_data.get("content")
        or report_data.get("report")
        or report_data.get("markdown")
        or report_data.get("text")
    )
    if extracted_text and isinstance(extracted_text, str):
        st.markdown(extracted_text)
        return

    for key, content in report_data.items():
        section_title = key.replace("_", " ").title()
        st.markdown(f"## {section_title}")

        if "claim" in key.lower() and isinstance(content, list):
            render_chatbot_matrix(content)
        elif "synergy" in key.lower() and isinstance(content, list):
            render_synergy_matrix(content)
        elif "limitation" in key.lower() and isinstance(content, list):
            render_limitations(content)
        elif isinstance(content, list):
            for item in content:
                if isinstance(item, dict):
                    st.write(item)
                else:
                    st.markdown(f"* {item}")
        elif isinstance(content, dict):
            st.write(content)
        else:
            st.markdown(str(content))

        st.markdown("---")


st.set_page_config(
    page_title="LONTAR | Literature Pipeline Engine",
    page_icon=":microscope:",
    layout="wide",
)
load_dotenv()
hydrate_environment_from_streamlit_secrets()

st.title("LONTAR Literature Pipeline Engine")
st.caption("Research ingestion, vectorization, and multi-source report synthesis.")
st.markdown("---")

if "ingested_records" not in st.session_state:
    st.session_state.ingested_records = []
if "embedding_success" not in st.session_state:
    st.session_state.embedding_success = False
if "active_collection" not in st.session_state:
    st.session_state.active_collection = "lontar_ingredient_research_v4"
if "active_cohere_model" not in st.session_state:
    st.session_state.active_cohere_model = "embed-v4.0"

tab1, tab2, tab3 = st.tabs(
    ["1. Ingestion Stage", "2. Filter & Embed", "3. Synthesize Report"]
)

with tab1:
    st.header("Literature Source Ingestion")

    col1, col2 = st.columns(2)
    with col1:
        ingredients_input = st.text_input(
            "Target Research Ingredients (comma separated)",
            "Curcuma longa, Piper nigrum",
        )
        per_page = st.number_input("Records per engine page", min_value=1, max_value=50, value=10)

    with col2:
        openalex = st.slider("OpenAlex Pages", 0, 5, 1)
        semantic = st.slider("Semantic Scholar Pages", 0, 5, 1)
        pubmed = st.slider("PubMed Pages", 0, 5, 1)
        core = st.slider("Core Pages", 0, 5, 0)

    if st.button("Run Ingestion Pipeline", type="primary"):
        ingredients_list = clean_ingredients(ingredients_input)

        if not ingredients_list:
            st.error("Please enter at least one ingredient.")
        else:
            with st.spinner("Fetching matching literature records across active repositories..."):
                try:
                    records = asyncio.run(
                        ingest_ingredients(
                            ingredients=ingredients_list,
                            openalex_pages=openalex,
                            semantic_scholar_pages=semantic,
                            pubmed_pages=pubmed,
                            core_pages=core,
                            per_page=per_page,
                        )
                    )
                    st.session_state.ingested_records = records_to_dicts(records)
                    st.session_state.embedding_success = False
                    st.success(
                        f"Successfully ingested {len(st.session_state.ingested_records)} source records."
                    )
                except Exception as exc:
                    st.error(f"Ingestion failed: {exc}")

    if st.session_state.ingested_records:
        st.markdown("---")
        st.subheader("Found Academic Literature Records")
        st.markdown("Click on any document below to inspect authors, abstracts, and identifiers.")

        for record in st.session_state.ingested_records:
            title = (record.get("title") or "Untitled Document").strip()
            year = record.get("year")
            source = (record.get("source") or "Unknown").upper()
            citations = record.get("citation_count")

            header_str = title
            if year:
                header_str += f" ({year})"

            with st.expander(header_str):
                c1, c2 = st.columns([3, 1])
                with c1:
                    authors = record.get("authors", [])
                    authors_str = ", ".join(authors) if authors else "Unknown Authors"
                    st.markdown(f"**Authors:** *{authors_str}*")

                    abstract = record.get("abstract")
                    st.markdown("**Abstract:**")
                    st.write(
                        abstract
                        if abstract
                        else "*No abstract summary was provided by the repository engine.*"
                    )

                with c2:
                    st.markdown(f"**Data Source:** `{source}`")
                    if citations is not None:
                        st.markdown(f"**Citation Count:** `{citations}`")
                    if record.get("doi"):
                        st.markdown(f"**DOI:** `{record.get('doi')}`")

                    if record.get("url"):
                        st.link_button(
                            "Open Publisher Page",
                            record.get("url"),
                            use_container_width=True,
                        )

with tab2:
    st.header("Guardrail Filtration & Local Vector Upsert")

    if not st.session_state.ingested_records:
        st.warning("Please complete the Ingestion Stage first to collect raw records.")
    else:
        col1, col2 = st.columns(2)
        with col1:
            min_citations = st.number_input("Minimum Citation Guardrail", min_value=0, value=0)
            recent_year = st.number_input(
                "Minimum Publication Year Guardrail",
                min_value=1900,
                max_value=2026,
                value=2018,
            )
        with col2:
            cohere_model = st.selectbox(
                "Cohere Vector Model",
                ["embed-v4.0", "embed-multilingual-v3.0"],
                index=0,
            )
            collection_name = st.text_input(
                "Chroma Vector Target Collection",
                st.session_state.active_collection,
            )

        if st.button("Process & Generate Embeddings", type="primary"):
            with st.spinner("Executing filtering criteria and updating vector embeddings via Cohere..."):
                try:
                    filtered = filter_records(
                        st.session_state.ingested_records,
                        min_citations=min_citations,
                        recent_year=recent_year,
                    )
                    chunks = build_chunks(filtered)
                    if not chunks:
                        st.error("No records passed the current filters with usable abstracts.")
                        st.stop()

                    embed_and_upsert(
                        chunks=chunks,
                        chroma_path=CHROMA_PATH,
                        collection_name=collection_name,
                        cohere_model=cohere_model,
                        batch_size=64,
                    )
                    st.session_state.embedding_success = True
                    st.session_state.active_collection = collection_name
                    st.session_state.active_cohere_model = cohere_model

                    st.balloons()
                    st.success("Vector space processing complete.")

                    st.markdown("### Vector Space Matrix Summary")
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Total Raw Inputs Found", len(st.session_state.ingested_records))
                    m2.metric("Passed Guardrail Filters", len(filtered))
                    m3.metric("Chroma DB Status", "Synchronized")

                    st.info(
                        "Pipeline feedback: successfully processed guardrails and updated "
                        "the local vector collection store."
                    )
                except Exception as exc:
                    st.error(f"Embedding failed: {exc}")

with tab3:
    st.header("Multi-Source Research Report Synthesis")

    synth_ingredients = st.text_input("Query Ingredients for Knowledge Retrieval", "Curcuma longa")

    col1, col2 = st.columns(2)
    with col1:
        synth_collection = st.text_input(
            "Retrieval Collection",
            st.session_state.active_collection,
        )
        synth_cohere_model = st.selectbox(
            "Retrieval Cohere Model",
            ["embed-v4.0", "embed-multilingual-v3.0"],
            index=0 if st.session_state.active_cohere_model == "embed-v4.0" else 1,
        )
        top_k = st.slider("Source Documents to Retrieve", 1, 25, 12)
    with col2:
        llm_provider = st.selectbox("Generation Provider", ["groq", "gemini"])
        groq_model = st.text_input("Groq Model", "llama-3.3-70b-versatile")
        gemini_model = st.text_input("Gemini Model", "gemini-2.0-flash")

    if st.button("Generate Comprehensive Report", type="primary"):
        ingredients_list = clean_ingredients(synth_ingredients)

        if not ingredients_list:
            st.error("Please enter at least one ingredient.")
        else:
            with st.spinner("Querying vector space and synthesizing report..."):
                try:
                    sources = retrieve_sources(
                        ingredients=ingredients_list,
                        chroma_path=CHROMA_PATH,
                        collection_name=synth_collection,
                        cohere_model=synth_cohere_model,
                        top_k=top_k,
                    )
                    if not sources:
                        st.error(
                            "No source reference literature found in the vector collection "
                            "for this query."
                        )
                        st.stop()

                    prompt = build_prompt(ingredients_list, sources)
                    if llm_provider == "groq":
                        report = generate_report_groq(prompt, model_name=groq_model)
                    else:
                        report = generate_report(
                            prompt,
                            model_name=gemini_model,
                            max_retries=3,
                            backoff_seconds=2.0,
                        )

                    st.success("Report successfully generated.")
                    st.markdown("---")
                    st.header("AI Comprehensive Research Report")
                    render_report(report.model_dump())
                except Exception as exc:
                    st.error(f"Synthesis failed: {exc}")
