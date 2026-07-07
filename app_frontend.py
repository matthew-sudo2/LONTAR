import asyncio
import logging
import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from streamlit.errors import StreamlitSecretNotFoundError

from src.config import (
    CHROMA_PATH,
    COHERE_EMBED_MODEL,
    COHERE_EMBED_MODELS,
    DEFAULT_BATCH_SIZE,
    DEFAULT_EMBED_FIELD,
    DEFAULT_FOCUS,
    DEFAULT_TOP_K,
    FOCUS_OPTIONS,
    GROQ_MODELS,
    INGESTION_PAGE_WARNING_THRESHOLD,
    LONTAR_COLLECTION,
    MIN_CITATIONS,
    RECENT_YEAR,
    SUMMARY_GROQ_MODELS,
    DEFAULT_SUMMARY_MODEL,
)
from src.ingestion import ingest_ingredients, records_to_dicts
from src.phase2 import (
    apply_summaries_to_records,
    build_chunks,
    embed_and_upsert,
    filter_records,
    summarize_for_health_benefits,
)
from src.phase3 import (
    build_prompt,
    generate_report_groq,
    retrieve_sources,
)
from src.relevance import triage_records, validate_query_ingredients

logger = logging.getLogger(__name__)

SECRET_KEYS = [
    "GEMINI_API_KEY",
    "SEMANTIC_SCHOLAR_API_KEY",
    "COHERE_API_KEY",
    "GROQ_API_KEY",
    "CORE_API_KEY",
    "OPENALEX_EMAIL",
]


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


def direction_badge(direction: str | None) -> str:
    return {
        "benefit": "🟢 Benefit",
        "risk": "🔴 Risk",
        "neutral": "⚪ Neutral",
        "unrelated": "⚫ Unrelated",
    }.get(direction or "", "")


def render_record_body(record: dict) -> None:
    if record.get("benefit_summary"):
        badge = direction_badge(record.get("direction"))
        st.markdown(f"**Health-Benefit Summary** {badge}")
        st.write(record.get("benefit_summary"))
        if record.get("mechanism"):
            st.markdown(f"**Mechanism:** {record.get('mechanism')}")
        if record.get("population"):
            st.markdown(f"**Population:** {record.get('population')}")
        if record.get("confidence"):
            st.markdown(f"**Confidence:** `{record.get('confidence')}`")
        with st.expander("Show raw abstract"):
            st.write(record.get("abstract") or "*No abstract available.*")
    else:
        abstract = record.get("abstract")
        st.markdown("**Abstract:**")
        st.write(
            abstract
            if abstract
            else "*No abstract summary was provided by the repository engine.*"
        )


def render_session_context() -> None:
    ingredients = st.session_state.active_ingredients
    focus = st.session_state.active_focus
    if ingredients:
        st.info(
            f"**Active session:** ingredients = `{', '.join(ingredients)}` · "
            f"focus = `{focus}` · collection = `{st.session_state.active_collection}` · "
            f"embed model = `{st.session_state.active_cohere_model}`"
        )
    else:
        st.caption("No ingestion session yet. Complete Tab 1 to set active ingredients and focus.")


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


def render_retrieval_audit(sources: list[dict], rejected: list[dict]) -> None:
    st.subheader("Retrieval Audit")
    st.caption(
        "Sources below passed vector similarity and relevance re-filtering. "
        "Rejected candidates are shown for transparency."
    )

    for source in sources:
        title = (source.get("title") or "Untitled").strip()
        badge = direction_badge(source.get("direction"))
        label = f"Accepted: {title} {badge}".strip()
        with st.expander(label):
            st.markdown(f"**Relevance score:** `{source.get('relevance_score')}`")
            if source.get("benefit_summary"):
                st.markdown(f"**Benefit summary:** {source.get('benefit_summary')}")
            if source.get("mechanism"):
                st.markdown(f"**Mechanism:** {source.get('mechanism')}")
            if source.get("confidence"):
                st.markdown(f"**Confidence:** `{source.get('confidence')}`")
            for reason in source.get("relevance_reasons") or []:
                st.markdown(f"- {reason}")

    if rejected:
        with st.expander(f"Rejected at retrieval ({len(rejected)})"):
            for source in rejected[:20]:
                title = (source.get("title") or "Untitled").strip()
                st.markdown(f"**{title}**")
                st.markdown(f"Score: `{source.get('relevance_score')}`")
                for reason in source.get("relevance_reject_reasons") or []:
                    st.markdown(f"- {reason}")
                signals = source.get("relevance_reasons") or []
                if signals:
                    st.caption("Signals: " + "; ".join(signals[:3]))
                st.markdown("---")


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
if "rejected_records" not in st.session_state:
    st.session_state.rejected_records = []
if "active_ingredients" not in st.session_state:
    st.session_state.active_ingredients = []
if "active_focus" not in st.session_state:
    st.session_state.active_focus = DEFAULT_FOCUS
if "embedding_success" not in st.session_state:
    st.session_state.embedding_success = False
if "active_collection" not in st.session_state:
    st.session_state.active_collection = LONTAR_COLLECTION
if "active_cohere_model" not in st.session_state:
    st.session_state.active_cohere_model = COHERE_EMBED_MODEL

tab1, tab2, tab3 = st.tabs(
    ["1. Ingestion Stage", "2. Filter & Embed", "3. Synthesize Report"]
)

with tab1:
    st.header("Literature Source Ingestion")

    col1, col2 = st.columns(2)
    with col1:
        default_ingredients = (
            ", ".join(st.session_state.active_ingredients)
            if st.session_state.active_ingredients
            else "Curcuma longa, Piper nigrum"
        )
        ingredients_input = st.text_input(
            "Target Research Ingredients (comma separated)",
            default_ingredients,
        )
        focus_index = (
            list(FOCUS_OPTIONS).index(st.session_state.active_focus)
            if st.session_state.active_focus in FOCUS_OPTIONS
            else 0
        )
        research_focus = st.selectbox(
            "Research Focus",
            list(FOCUS_OPTIONS),
            index=focus_index,
        )
        if research_focus == "Broad/Generic (No Filter)":
            st.caption(
                "Broad/Generic applies no focus-profile keyword filter. "
                "Only ingredient-term matching and citation/year guardrails remain."
            )
        per_page = st.number_input("Records per engine page", min_value=1, max_value=50, value=10)

    with col2:
        openalex = st.slider("OpenAlex Pages", 0, 5, 1)
        semantic = st.slider("Semantic Scholar Pages", 0, 5, 1)
        pubmed = st.slider("PubMed Pages", 0, 5, 1)
        core = st.slider("Core Pages", 0, 5, 0)

    total_page_fetches = openalex + semantic + pubmed + core
    if total_page_fetches > INGESTION_PAGE_WARNING_THRESHOLD:
        st.warning(
            f"High API load: {total_page_fetches} page fetches per source engine "
            f"(threshold {INGESTION_PAGE_WARNING_THRESHOLD}). "
            "This can hit rate limits and increase latency."
        )

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
                            focus=research_focus,
                            openalex_pages=openalex,
                            semantic_scholar_pages=semantic,
                            pubmed_pages=pubmed,
                            core_pages=core,
                            per_page=per_page,
                        )
                    )
                    raw_records = records_to_dicts(records)
                    accepted, rejected = triage_records(
                        raw_records,
                        ingredients=ingredients_list,
                        focus=research_focus,
                    )
                    st.session_state.ingested_records = accepted
                    st.session_state.rejected_records = rejected
                    st.session_state.active_ingredients = ingredients_list
                    st.session_state.active_focus = research_focus
                    st.session_state.embedding_success = False
                    st.success(
                        f"Accepted {len(accepted)} relevant records from {len(raw_records)} raw results."
                    )
                    if rejected:
                        st.info(
                            f"Held back {len(rejected)} low-relevance records. Expand the rejected section below to audit why."
                        )
                except Exception as exc:
                    logger.exception("Ingestion failed")
                    st.error(f"Ingestion failed: {exc}")

    if st.session_state.ingested_records:
        st.markdown("---")
        st.subheader("Accepted Academic Literature Records")
        st.markdown("Click on any accepted document below to inspect authors, abstracts, relevance signals, and identifiers.")

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
                    render_record_body(record)

                with c2:
                    st.markdown(f"**Data Source:** `{source}`")
                    if citations is not None:
                        st.markdown(f"**Citation Count:** `{citations}`")
                    if record.get("doi"):
                        st.markdown(f"**DOI:** `{record.get('doi')}`")
                    if record.get("relevance_score") is not None:
                        st.markdown(f"**Relevance Score:** `{record.get('relevance_score')}`")
                    reasons = record.get("relevance_reasons") or []
                    if reasons:
                        st.markdown("**Relevance Signals:**")
                        for reason in reasons[:3]:
                            st.markdown(f"- {reason}")

                    if record.get("url"):
                        st.link_button(
                            "Open Publisher Page",
                            record.get("url"),
                            use_container_width=True,
                        )

    if st.session_state.rejected_records:
        st.markdown("---")
        with st.expander(
            f"Rejected or Low-Relevance Records ({len(st.session_state.rejected_records)})"
        ):
            for record in st.session_state.rejected_records:
                title = (record.get("title") or "Untitled Document").strip()
                year = record.get("year")
                label = f"{title} ({year})" if year else title
                st.markdown(f"**{label}**")
                st.markdown(f"Relevance score: `{record.get('relevance_score')}`")
                reject_reasons = record.get("relevance_reject_reasons") or []
                for reason in reject_reasons[:4]:
                    st.markdown(f"- {reason}")
                signals = record.get("relevance_reasons") or []
                if signals:
                    st.caption("Signals: " + "; ".join(signals[:3]))
                st.markdown("---")

with tab2:
    st.header("Guardrail Filtration & Local Vector Upsert")
    render_session_context()

    if not st.session_state.ingested_records:
        st.warning("Please complete the Ingestion Stage first to collect raw records.")
    else:
        col1, col2 = st.columns(2)
        with col1:
            min_citations = st.number_input(
                "Minimum Citation Guardrail",
                min_value=0,
                value=MIN_CITATIONS,
                help="Record passes if it meets this citation count OR the recent-year threshold.",
            )
            recent_year = st.number_input(
                "Minimum Publication Year Guardrail",
                min_value=1900,
                max_value=2026,
                value=RECENT_YEAR,
                help="Record passes if published in this year or later OR meets citation threshold.",
            )
        with col2:
            model_index = (
                list(COHERE_EMBED_MODELS).index(st.session_state.active_cohere_model)
                if st.session_state.active_cohere_model in COHERE_EMBED_MODELS
                else 0
            )
            cohere_model = st.selectbox(
                "Cohere Vector Model",
                list(COHERE_EMBED_MODELS),
                index=model_index,
            )
            collection_name = st.text_input(
                "Chroma Vector Target Collection",
                st.session_state.active_collection,
            )

        st.markdown("#### AI Health-Benefit Summarization")
        summarize_enabled = st.checkbox(
            "Summarize each source with an emphasis on health benefits (recommended)",
            value=True,
            help="Runs each accepted abstract through Groq to extract a health-benefit-focused "
            "summary, effect direction, mechanism, and confidence. The summary is embedded "
            "instead of the raw abstract, improving retrieval for health-benefit queries.",
        )
        drop_risk = False
        summary_model = DEFAULT_SUMMARY_MODEL
        if summarize_enabled:
            c1, c2 = st.columns(2)
            with c1:
                summary_model = st.selectbox(
                    "Summarization Model",
                    list(SUMMARY_GROQ_MODELS),
                    index=0,
                )
            with c2:
                drop_risk = st.checkbox(
                    "Exclude risk/unrelated findings from the vector store",
                    value=False,
                    help="Keeps risk-direction sources out of the health-claim matrix entirely.",
                )

        if st.button("Process & Generate Embeddings", type="primary"):
            with st.spinner("Executing filtering criteria and updating vector embeddings via Cohere..."):
                try:
                    filtered = filter_records(
                        st.session_state.ingested_records,
                        min_citations=min_citations,
                        recent_year=recent_year,
                        ingredients=st.session_state.active_ingredients,
                        focus=st.session_state.active_focus,
                    )

                    set_aside: list[dict] = []
                    if summarize_enabled:
                        with st.spinner("Summarizing sources for health-benefit emphasis..."):
                            filtered, set_aside = summarize_for_health_benefits(
                                filtered,
                                ingredients=st.session_state.active_ingredients,
                                summary_model=summary_model,
                                drop_risk_and_unrelated=drop_risk,
                            )
                        st.session_state.ingested_records = apply_summaries_to_records(
                            st.session_state.ingested_records,
                            filtered,
                        )

                    embed_field = DEFAULT_EMBED_FIELD if summarize_enabled else "abstract"
                    chunks = build_chunks(filtered, embed_field=embed_field)
                    if not chunks:
                        st.error("No records passed the current filters with usable abstracts.")
                        st.stop()

                    if set_aside:
                        st.info(
                            f"Summarizer set aside {len(set_aside)} risk/unrelated "
                            "records from the vector store."
                        )

                    embed_and_upsert(
                        chunks=chunks,
                        chroma_path=CHROMA_PATH,
                        collection_name=collection_name,
                        cohere_model=cohere_model,
                        batch_size=DEFAULT_BATCH_SIZE,
                    )
                    st.session_state.embedding_success = True
                    st.session_state.active_collection = collection_name
                    st.session_state.active_cohere_model = cohere_model

                    st.balloons()
                    st.success("Vector space processing complete.")

                    st.markdown("### Vector Space Matrix Summary")
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Accepted Ingestion Inputs", len(st.session_state.ingested_records))
                    m2.metric("Passed Guardrail Filters", len(filtered))
                    m3.metric("Chroma DB Status", "Synchronized")

                    st.info(
                        "Pipeline feedback: successfully processed guardrails and updated "
                        "the local vector collection store."
                    )
                except Exception as exc:
                    logger.exception("Embedding failed")
                    st.error(f"Embedding failed: {exc}")

with tab3:
    st.header("Multi-Source Research Report Synthesis")
    render_session_context()

    if not st.session_state.embedding_success:
        st.warning(
            "No embeddings found for this session. Complete Tab 2 before synthesizing a report."
        )

    embedded_default = (
        ", ".join(st.session_state.active_ingredients)
        if st.session_state.active_ingredients
        else "Curcuma longa"
    )
    synth_ingredients = st.text_input(
        "Query Ingredients for Knowledge Retrieval",
        embedded_default,
        help="Must match ingredients embedded in Tab 2 (aliases like turmeric/curcuma longa are accepted).",
    )

    col1, col2 = st.columns(2)
    with col1:
        synth_collection = st.selectbox(
            "Retrieval Collection",
            [st.session_state.active_collection],
            index=0,
        )
        model_index = (
            list(COHERE_EMBED_MODELS).index(st.session_state.active_cohere_model)
            if st.session_state.active_cohere_model in COHERE_EMBED_MODELS
            else 0
        )
        synth_cohere_model = st.selectbox(
            "Retrieval Cohere Model",
            list(COHERE_EMBED_MODELS),
            index=model_index,
        )
        if synth_cohere_model != st.session_state.active_cohere_model:
            st.error(
                f"Embedding model mismatch: Tab 2 used `{st.session_state.active_cohere_model}` "
                f"but retrieval is set to `{synth_cohere_model}`. Use the same model for both."
            )
    with col2:
        groq_model = st.selectbox(
            "Groq Model",
            list(GROQ_MODELS),
            index=0,
        )
        st.caption("Groq model IDs change over time; update GROQ_MODELS in src/config.py as needed.")
        top_k = st.slider("Source Documents to Retrieve", 1, 25, DEFAULT_TOP_K)

    if st.button("Generate Comprehensive Report", type="primary"):
        ingredients_list = clean_ingredients(synth_ingredients)

        if not ingredients_list:
            st.error("Please enter at least one ingredient.")
        elif not st.session_state.embedding_success:
            st.error("Complete Tab 2 (Filter & Embed) before generating a report.")
        elif synth_cohere_model != st.session_state.active_cohere_model:
            st.error("Fix the Cohere model mismatch before generating a report.")
        else:
            embedded = st.session_state.active_ingredients
            if embedded:
                matched, unmatched = validate_query_ingredients(ingredients_list, embedded)
                if unmatched:
                    st.error(
                        "Query ingredients not found in the embedded session set: "
                        f"`{', '.join(unmatched)}`. "
                        f"Embedded ingredients were: `{', '.join(embedded)}`. "
                        "Re-run Tab 1/2 with matching ingredients or adjust your query."
                    )
                    st.stop()

            with st.spinner("Querying vector space and synthesizing report..."):
                try:
                    retrieval = retrieve_sources(
                        ingredients=ingredients_list,
                        chroma_path=CHROMA_PATH,
                        collection_name=synth_collection,
                        cohere_model=synth_cohere_model,
                        top_k=top_k,
                        focus=st.session_state.active_focus,
                    )
                    if not retrieval.sources:
                        st.error(
                            "No source reference literature found in the vector collection "
                            "for this query. Check that Tab 2 used the same collection and "
                            f"embed model (`{st.session_state.active_cohere_model}`)."
                        )
                        if retrieval.rejected:
                            render_retrieval_audit([], retrieval.rejected)
                        st.stop()

                    render_retrieval_audit(retrieval.sources, retrieval.rejected)

                    prompt = build_prompt(ingredients_list, retrieval.sources)
                    report = generate_report_groq(prompt, model_name=groq_model)

                    st.success("Report successfully generated.")
                    st.markdown("---")
                    st.header("AI Comprehensive Research Report")
                    render_report(report.model_dump())
                except Exception as exc:
                    logger.exception("Synthesis failed")
                    st.error(f"Synthesis failed: {exc}")
