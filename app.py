import logging
import os
import streamlit as st
from typing import Dict, List, Optional

from src.config import settings
from src.document_loader import DocumentLoaderError, DocumentProcessor
from src.models import ChatResponse
from src.rag_engine import RAGEngine
from src.vector_store import EmbeddingModelMismatchError, VectorStoreManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Streamlit Page Configuration
st.set_page_config(
    page_title="RAG PDF Chatbot",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ------------------------------------------------------------------
# Cached Resources (Initialized once per application lifecycle)
# ------------------------------------------------------------------
@st.cache_resource(show_spinner="Initializing Vector Store & Embeddings...")
def get_vector_store_manager() -> VectorStoreManager:
    """Caches the ChromaDB vector store manager and embedding model."""
    return VectorStoreManager()


@st.cache_resource(show_spinner="Initializing RAG Engine...")
def get_rag_engine() -> RAGEngine:
    """Caches the RAG engine and LLM pipeline."""
    vstore = get_vector_store_manager()
    return RAGEngine(vstore)


# ------------------------------------------------------------------
# Session State Initialization
# ------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "processed_hashes" not in st.session_state:
    st.session_state.processed_hashes = set()


# ------------------------------------------------------------------
# Main UI Logic
# ------------------------------------------------------------------
def main():
    # Ensure data directories exist
    settings.ensure_directories()

    try:
        vstore = get_vector_store_manager()
        rag_engine = get_rag_engine()
    except EmbeddingModelMismatchError as e:
        st.error(f"⚠️ **Embedding Model Mismatch**: {e}")
        if st.button("Reset Vector Store to Fix Mismatch"):
            temp_vstore = VectorStoreManager()
            temp_vstore.reset()
            st.cache_resource.clear()
            st.rerun()
        return

    # ------------------------------------------------------------------
    # Sidebar: Document Management & Settings
    # ------------------------------------------------------------------
    with st.sidebar:
        st.title("🤖 RAG PDF Chatbot")
        st.markdown(
            "Upload PDFs and ask questions grounded strictly in their content with verifiable page citations."
        )

        st.divider()

        # System Status & Configuration
        st.caption(f"**LLM Model:** `{settings.gemini_model if settings.llm_provider == 'gemini' else settings.openai_model}`")
        st.caption(f"**Embedding:** `{settings.embedding_model}` ({settings.embedding_provider})")
        st.caption(f"**Similarity Threshold:** `{settings.relevance_threshold}`")

        # API Key Override (if not in .env)
        if settings.llm_provider == "gemini" and not settings.gemini_api_key:
            api_key_input = st.text_input("Enter Gemini API Key", type="password")
            if api_key_input:
                settings.gemini_api_key = api_key_input
                st.cache_resource.clear()
                st.rerun()

        st.divider()

        # PDF Upload Section
        st.subheader("📄 Upload Documents")
        uploaded_files = st.file_uploader(
            "Upload PDF files",
            type=["pdf"],
            accept_multiple_files=True,
            help=f"Maximum file size: {settings.max_file_size_mb} MB per document."
        )

        processor = DocumentProcessor()

        if uploaded_files:
            for uploaded_file in uploaded_files:
                file_bytes = uploaded_file.getvalue()
                filename = uploaded_file.name

                file_hash = processor.compute_sha256(file_bytes)

                # Skip if already processed in this session or in Chroma
                if file_hash in st.session_state.processed_hashes or vstore.is_file_indexed(file_hash):
                    st.session_state.processed_hashes.add(file_hash)
                    continue

                with st.spinner(f"Ingesting '{filename}'..."):
                    try:
                        chunks, doc_info = processor.process_pdf(file_bytes, filename)
                        vstore.upsert_chunks(chunks)
                        st.session_state.processed_hashes.add(file_hash)
                        st.success(f"✅ Ingested **{filename}** ({doc_info.page_count} pages, {len(chunks)} chunks)")
                    except DocumentLoaderError as e:
                        st.error(f"❌ Error with '{filename}': {e}")
                    except Exception as e:
                        st.error(f"❌ Unexpected error reading '{filename}': {e}")

        # Active Document Registry
        indexed_docs = vstore.get_indexed_documents()
        st.subheader(f"📚 Knowledge Base ({len(indexed_docs)} Docs)")

        if indexed_docs:
            for doc in indexed_docs:
                st.markdown(f"• **{doc.filename}** ({doc.page_count} pages, {doc.chunk_count} chunks)")

            st.divider()

            # Document Filter Selector (Filters by file_hash)
            doc_options = {"All Documents": None}
            for doc in indexed_docs:
                doc_options[f"{doc.filename} ({doc.page_count}p)"] = doc.file_hash

            selected_doc_label = st.selectbox(
                "🎯 Query Target:",
                options=list(doc_options.keys()),
                help="Filter questions to a specific uploaded document or search across all documents."
            )
            selected_file_hash = doc_options[selected_doc_label]
        else:
            st.info("No documents indexed yet. Upload a PDF above to begin.")
            selected_file_hash = None

        st.divider()

        # Management Controls
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Clear Chat", use_container_width=True):
                st.session_state.messages = []
                st.rerun()

        with col2:
            if st.button("Reset KB", use_container_width=True, help="Deletes all indexed documents from Chroma"):
                vstore.reset()
                st.session_state.processed_hashes = set()
                st.session_state.messages = []
                st.cache_resource.clear()
                st.success("Knowledge base reset.")
                st.rerun()

        st.caption(
            "🔒 *Security Note: PDF extracted text is treated as untrusted input. Grounding prompt enforces strict context boundaries.*"
        )

    # ------------------------------------------------------------------
    # Main Panel: Conversational Interface
    # ------------------------------------------------------------------
    st.header("💬 Document Intelligence Chat")

    if not indexed_docs:
        st.markdown(
            """
            ### Welcome to the RAG PDF Chatbot! 👋
            To get started:
            1. **Upload one or more PDFs** in the sidebar.
            2. Choose whether to query **All Documents** or focus on a specific document.
            3. Ask questions in natural language.
            
            *Every answer is grounded strictly in the source text and includes precise page-level citations.*
            """
        )
        return

    # Render Chat History
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            citations = msg.get("citations", [])
            if citations:
                with st.expander(f"📚 View {len(citations)} Source Citation(s)"):
                    for idx, c in enumerate(citations, 1):
                        st.markdown(f"**Citation {idx}:** `{c['file_name']}` — **Page {c['page_number']}**")
                        st.caption(f"> \"{c['snippet']}\"")

    # Chat Input Box
    if user_prompt := st.chat_input("Ask a question about the uploaded document(s)..."):
        # Append and display user message
        st.session_state.messages.append({"role": "user", "content": user_prompt})
        with st.chat_message("user"):
            st.markdown(user_prompt)

        # Generate Grounded RAG Answer
        with st.chat_message("assistant"):
            with st.spinner("Retrieving relevant context and synthesizing answer..."):
                response: ChatResponse = rag_engine.answer_query(
                    query=user_prompt,
                    chat_history=st.session_state.messages[:-1],  # Exclude current prompt
                    file_hash_filter=selected_file_hash
                )

                st.markdown(response.answer)

                citations_data = []
                if response.citations and not response.is_fallback:
                    with st.expander(f"📚 View {len(response.citations)} Source Citation(s)"):
                        for idx, c in enumerate(response.citations, 1):
                            st.markdown(f"**Citation {idx}:** `{c.file_name}` — **Page {c.page_number}**")
                            st.caption(f"> \"{c.snippet}\"")
                            citations_data.append({
                                "file_name": c.file_name,
                                "page_number": c.page_number,
                                "file_hash": c.file_hash,
                                "snippet": c.snippet
                            })

        # Persist assistant response with citations
        st.session_state.messages.append({
            "role": "assistant",
            "content": response.answer,
            "citations": citations_data
        })


if __name__ == "__main__":
    main()
