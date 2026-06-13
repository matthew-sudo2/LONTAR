import streamlit as st
import requests

# 1. Page Configuration Setup
st.set_page_config(
    page_title="LONTAR | Literature Pipeline Engine",
    page_icon="🔬",
    layout="wide"
)

BACKEND_URL = "http://127.0.0.1:8080" # Update this if your Uvicorn port changes

st.title("🔬 LONTAR Literature Pipeline Engine")
st.caption("Synchronous functional research ingestion, matrix vectorization, and multi-source report synthesis.")
st.markdown("---")

# 2. Maintain state across tab refreshes
if "ingested_records" not in st.session_state:
    st.session_state.ingested_records = []
if "embedding_success" not in st.session_state:
    st.session_state.embedding_success = False

# 3. Create clean layout tabs matching your backend stages
tab1, tab2, tab3 = st.tabs(["📥 1. Ingestion Stage", "🧠 2. Filter & Embed", "📝 3. Synthesize Report"])

# =====================================================================
# TAB 1: INGESTION STAGE
# =====================================================================
with tab1:
    st.header("Literature Source Ingestion")
    
    col1, col2 = st.columns(2)
    with col1:
        ingredients_input = st.text_input("Target Research Ingredients (comma separated)", "Curcuma longa, Piper nigrum")
        per_page = st.number_input("Records per engine page", min_value=1, max_value=50, value=10)
    
    with col2:
        openalex = st.slider("OpenAlex Pages", 0, 5, 1)
        semantic = st.slider("Semantic Scholar Pages", 0, 5, 1)
        pubmed = st.slider("PubMed Pages", 0, 5, 1)
        core = st.slider("Core Pages", 0, 5, 0)

    if st.button("Run Ingestion Pipeline", type="primary"):
        ingredients_list = [i.strip() for i in ingredients_input.split(",")]
        
        payload = {
            "ingredients": ingredients_list,
            "openalex_pages": openalex,
            "semantic_pages": semantic,
            "pubmed_pages": pubmed,
            "core_pages": core,
            "per_page": per_page
        }
        
        with st.spinner("Fetching matching literature records across active database repositories..."):
            try:
                response = requests.post(f"{BACKEND_URL}/api/v1/ingest", json=payload)
                if response.status_code == 200:
                    st.session_state.ingested_records = response.json()
                    st.success(f"Successfully ingested {len(st.session_state.ingested_records)} source records!")
                else:
                    st.error(f"Ingestion failed: {response.text}")
            except Exception as e:
                st.error(f"Could not connect to backend server: {str(e)}")

    if st.session_state.ingested_records:
        st.markdown("---")
        st.subheader("📄 Found Academic Literature Records")
        st.markdown("Click on any document below to inspect authors, abstracts, and identifiers.")
        
        for idx, record in enumerate(st.session_state.ingested_records):
            title = record.get("title", "Untitled Document").strip()
            year = record.get("year")
            source = record.get("source", "Unknown").upper()
            citations = record.get("citation_count")
            
            header_str = f"📚 {title}"
            if year:
                header_str += f" ({year})"
            
            with st.expander(header_str):
                c1, c2 = st.columns([3, 1])
                with c1:
                    authors = record.get("authors", [])
                    authors_str = ", ".join(authors) if authors else "Unknown Authors"
                    st.markdown(f"**✍️ Authors:** *{authors_str}*")
                    
                    abstract = record.get("abstract")
                    st.markdown("**📖 Abstract:**")
                    st.write(abstract if abstract else "*No abstract summary was provided by the repository engine.*")
                
                with c2:
                    st.markdown(f"**🔌 Data Source:** `{source}`")
                    if citations is not None:
                        st.markdown(f"**📈 Citations Count:** `{citations}`")
                    if record.get("doi"):
                        st.markdown(f"**🆔 DOI:** `{record.get('doi')}`")
                    
                    if record.get("url"):
                        st.link_button("🌐 Open Publisher Page", record.get("url"), use_container_width=True)

# =====================================================================
# TAB 2: FILTER & EMBED
# =====================================================================
with tab2:
    st.header("Guardrail Filtration & Local Vector Upsert")
    
    if not st.session_state.ingested_records:
        st.warning("⚠️ Please complete the Ingestion Stage first to collect raw records.")
    else:
        col1, col2 = st.columns(2)
        with col1:
            min_citations = st.number_input("Minimum Citation Guardrail", min_value=0, value=0)
            recent_year = st.number_input("Minimum Publication Year Guardrail", min_value=1900, max_value=2026, value=2018)
        with col2:
            cohere_model = st.selectbox("Cohere Vector Model", ["embed-v4.0"])
            collection_name = st.text_input("Chroma Vector Target Collection", "lontar_ingredient_research_v4")

        if st.button("Process & Generate Embeddings", type="primary"):
            payload = {
                "records": st.session_state.ingested_records,
                "min_citations": min_citations,
                "recent_year": recent_year,
                "chroma_path": "data/chroma",
                "collection": collection_name,
                "cohere_model": cohere_model,
                "batch_size": 64
            }
            
            with st.spinner("Executing filtering criteria and updating vector embeddings via Cohere..."):
                try:
                    response = requests.post(f"{BACKEND_URL}/api/v1/filter-embed", json=payload)
                    if response.status_code == 200:
                        st.session_state.embedding_success = True
                        res_data = response.json()
                        
                        st.balloons()
                        st.success("Vector Space processing execution complete!")
                        
                        st.markdown("### 📊 Vector Space Matrix Summary")
                        m1, m2, m3 = st.columns(3)
                        m1.metric(label="Total Raw Inputs Found", value=res_data.get("input_count", 0))
                        m2.metric(label="Passed Guardrail Filters", value=res_data.get("filtered_count", 0))
                        m3.metric(label="Chroma DB Status", value="Synchronized ✅")
                        
                        st.info(f"💬 **Server Feedback:** {res_data.get('message', '')}")
                    else:
                        st.error(f"Embedding failed: {response.text}")
                except Exception as e:
                    st.error(f"Backend network connection failure: {str(e)}")

# =====================================================================
# TAB 3: SYNTHESIZE REPORT
# =====================================================================
with tab3:
    st.header("Multi-Source Research Report Synthesis")
    
    synth_ingredients = st.text_input("Query Ingredients for Knowledge Retrieval", "Curcuma longa")
    
    if st.button("Generate Comprehensive Report", type="primary"):
        ingredients_list = [i.strip() for i in synth_ingredients.split(",")]
        
        payload = {
            "ingredients": ingredients_list
        }
        
        with st.spinner("Querying vector space and synthesizing executive summary over Groq..."):
            try:
                response = requests.post(f"{BACKEND_URL}/api/v1/synthesize", json=payload)
                if response.status_code == 200:
                    report_data = response.json()
                    st.success("Report successfully generated!")
                    
                    st.markdown("---")
                    st.header("📋 AI Comprehensive Research Report")
                    
                    # 🛠️ HELPER 1: Chatbot Renderer for Health Claim Matrix
                    def render_chatbot_matrix(matrix_list):
                        for entry in matrix_list:
                            if isinstance(entry, dict) and "ingredient" in entry:
                                ingredient = entry.get("ingredient", "Unknown Ingredient")
                                claim = entry.get("claim", "No compiled health claim description.")
                                dois = entry.get("evidence_dois", [])
                                titles = entry.get("evidence_titles", [])
                                
                                with st.chat_message("assistant", avatar="🔬"):
                                    st.markdown(f"### 🌿 **Ingredient:** {ingredient}")
                                    st.markdown(f"🧬 **Identified Health Claim:** `{claim}`")
                                    
                                    if titles:
                                        st.markdown("**📚 Supporting Academic Evidence:**")
                                        for idx, title in enumerate(titles):
                                            doi_code = dois[idx] if idx < len(dois) else None
                                            if doi_code:
                                                st.markdown(f"{idx + 1}. *{title}* — [🌐 View Study (DOI: {doi_code})](https://doi.org/{doi_code})")
                                            else:
                                                st.markdown(f"{idx + 1}. *{title}*")
                                    elif dois:
                                        st.markdown("**📚 Supporting Academic Evidence (DOIs):**")
                                        for idx, doi_code in enumerate(dois):
                                            st.markdown(f"{idx + 1}. [🌐 View Study Document (DOI: {doi_code})](https://doi.org/{doi_code})")
                                    else:
                                        st.markdown("⚠️ *No explicit publication citations were bound to this claim.*")

                    # 🛠️ HELPER 2: Chatbot Renderer for Synergy Coefficients
                    def render_synergy_matrix(synergy_list):
                        for entry in synergy_list:
                            if isinstance(entry, dict):
                                ingredients = entry.get("ingredients", [])
                                formula_title = " + ".join(ingredients) if ingredients else "Unknown Blend"
                                mechanism = entry.get("mechanism", "No dynamic interactive mechanism description provided.")
                                dois = entry.get("evidence_dois", [])
                                
                                with st.chat_message("assistant", avatar="⚡"):
                                    st.markdown(f"### 🧪 **Synergistic Combo:** {formula_title}")
                                    st.markdown(f"⚙️ **Biological Mechanism:** {mechanism}")
                                    
                                    if dois:
                                        st.markdown("**📚 Evidence Links:**")
                                        for idx, doi in enumerate(dois):
                                            st.markdown(f"{idx + 1}. [🌐 Open Source Publication (DOI: {doi})](https://doi.org/{doi})")
                            else:
                                st.write(entry)

                    # 🛠️ HELPER 3: Chatbot Renderer for Limitations
                    def render_limitations(limitations_list):
                        with st.chat_message("assistant", avatar="⚠️"):
                            st.markdown("### 🛑 **Identified Research Gaps & Limitations**")
                            st.markdown("The pipeline flagged the following functional limits or constraints in current literature structures:")
                            for idx, item in enumerate(limitations_list):
                                st.markdown(f"* **Gap {idx+1}:** {item}")

                    # -----------------------------------------------------------------
                    # ROUTER: Dynamically check layout structures returning from Backend
                    # -----------------------------------------------------------------
                    if isinstance(report_data, list):
                        st.subheader("🔍 Analysis Results")
                        render_chatbot_matrix(report_data)
                        
                    elif isinstance(report_data, str):
                        st.markdown(report_data)
                        
                    elif isinstance(report_data, dict):
                        # Attempt to extract direct text payload fields if present
                        extracted_text = (
                            report_data.get("content") or 
                            report_data.get("report") or 
                            report_data.get("markdown") or 
                            report_data.get("text")
                        )
                        
                        if extracted_text and isinstance(extracted_text, str):
                            st.markdown(extracted_text)
                        else:
                            # Loop over structural dictionary blocks recursively
                            for key, content in report_data.items():
                                section_title = key.replace("_", " ").title()
                                st.markdown(f"## 🔍 {section_title}")
                                
                                # Route A: Is it the Health Claims block?
                                if "claim" in key.lower() and isinstance(content, list):
                                    render_chatbot_matrix(content)
                                    
                                # Route B: Is it the Synergy Coefficients block?
                                elif "synergy" in key.lower() and isinstance(content, list):
                                    render_synergy_matrix(content)
                                    
                                # Route C: Is it the Limitations block?
                                elif "limitation" in key.lower() and isinstance(content, list):
                                    render_limitations(content)
                                    
                                # General catch-all routing
                                elif isinstance(content, list):
                                    for item in content:
                                        st.markdown(f"* {item}")
                                elif isinstance(content, dict):
                                    st.write(content)
                                else:
                                    st.markdown(content)
                                    
                                st.markdown("---") # Visual spacing break between sections
                else:
                    st.error(f"Synthesis failed: {response.text}")
            except Exception as e:
                st.error(f"Backend connection error: {str(e)}")