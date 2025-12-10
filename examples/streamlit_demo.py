"""
Streamlit RAG Demo

Interactive web interface for the PolyRAG pipeline.

Run with: streamlit run streamlit_demo.py
"""

import streamlit as st
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from polyrag.interface.builder import PipelineBuilder
from polyrag.interface.factory import AdapterFactory


# Page config
st.set_page_config(
    page_title="PolyRAG Demo",
    page_icon="🔍",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1rem;
    }
    .stTextInput > div > div > input {
        font-size: 1.1rem;
    }
    .source-card {
        background: #f0f2f6;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_pipeline():
    """Initialize and cache the RAG pipeline."""
    try:
        llm = AdapterFactory.create_llm("ollama", model="llama3.2")
        embedding = AdapterFactory.create_embedding("fastembed")
        vector_store = AdapterFactory.create_vector_store("qdrant")
        loader = AdapterFactory.create_document_loader("text")
        chunker = AdapterFactory.create_chunker("fixed_size", chunk_size=500, chunk_overlap=50)

        pipeline = (
            PipelineBuilder()
            .with_llm(llm)
            .with_embedding(embedding)
            .with_vector_store(vector_store)
            .with_document_loader(loader)
            .with_chunker(chunker)
            .with_collection_name("streamlit_demo")
            .build()
        )
        return pipeline, None
    except Exception as e:
        return None, str(e)


def main():
    # Header
    st.markdown('<p class="main-header">🔍 PolyRAG Demo</p>', unsafe_allow_html=True)
    st.markdown("*Modular RAG Framework - Interactive Demo*")
    
    # Sidebar for settings
    with st.sidebar:
        st.header("⚙️ Settings")
        
        top_k = st.slider("Number of chunks to retrieve", 1, 10, 5)
        
        st.divider()
        
        st.header("📁 Document Upload")
        uploaded_file = st.file_uploader("Upload a text file", type=["txt", "md"])
        
        if uploaded_file is not None:
            # Save uploaded file
            file_path = f"uploaded_{uploaded_file.name}"
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getvalue())
            
            if st.button("📥 Ingest Document"):
                pipeline, error = get_pipeline()
                if error:
                    st.error(f"Pipeline error: {error}")
                else:
                    with st.spinner("Ingesting document..."):
                        try:
                            num_chunks = pipeline.ingest(file_path)
                            st.success(f"✅ Ingested {num_chunks} chunks!")
                        except Exception as e:
                            st.error(f"Error: {e}")
                    # Clean up
                    if os.path.exists(file_path):
                        os.remove(file_path)

    # Main content
    st.header("💬 Ask a Question")
    
    # Check pipeline status
    pipeline, error = get_pipeline()
    
    if error:
        st.error(f"⚠️ Could not initialize pipeline: {error}")
        st.info("Make sure Ollama and Qdrant are running.")
        return
    
    # Query input
    question = st.text_input(
        "Enter your question:",
        placeholder="What is the main topic of the document?"
    )
    
    col1, col2 = st.columns([1, 5])
    with col1:
        search_button = st.button("🔍 Search", type="primary")
    with col2:
        show_sources = st.checkbox("Show sources", value=True)
    
    if search_button and question:
        with st.spinner("Searching and generating answer..."):
            try:
                # Get retrieval results first
                results = pipeline.get_retrieval_results(question, top_k=top_k)
                
                # Generate answer with streaming
                st.subheader("📝 Answer")
                answer_placeholder = st.empty()
                full_answer = ""
                
                for chunk in pipeline.query_stream(question, top_k=top_k):
                    full_answer += chunk
                    answer_placeholder.markdown(full_answer + "▌")
                
                answer_placeholder.markdown(full_answer)
                
                # Show sources
                if show_sources and results:
                    st.subheader("📚 Sources")
                    for i, result in enumerate(results, 1):
                        with st.expander(f"Source {i} - Score: {result.score:.3f}"):
                            st.markdown(f"**File:** {result.chunk.metadata.get('file_name', 'Unknown')}")
                            st.markdown("**Content:**")
                            st.code(result.chunk.content, language=None)
                            
            except Exception as e:
                st.error(f"Error: {e}")
    
    # Footer
    st.divider()
    st.markdown("*Built with PolyRAG Framework*")


if __name__ == "__main__":
    main()
