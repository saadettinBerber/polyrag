"""
PolyRAG Final Demo

Comprehensive demonstration of all PolyRAG capabilities:
- Text and PDF document loading
- Vector and graph storage
- Multiple embedding options (FastEmbed, CLIP, ColBERT, ColPali)
- Hybrid retrieval
- Streaming LLM responses

Prerequisites:
- Ollama running (http://localhost:11434)
- Qdrant running (http://localhost:6333)
- Optional: Neo4j for graph features
"""

import os
import sys

# Ensure polyrag is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from polyrag.interface import PipelineBuilder, AdapterFactory

def print_header(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60)

def main():
    print_header("PolyRAG Framework - Final Demo")
    
    print("\n📦 Available Components:")
    print("-" * 40)
    
    # LLM options
    print("\n🤖 LLM Adapters:")
    print("   • OllamaAdapter - Local LLMs with vision support")
    
    # Embedding options
    print("\n🔢 Embedding Adapters:")
    print("   • FastEmbedAdapter - Fast local text embeddings")
    print("   • CLIPAdapter - Multimodal (text + image) embeddings")
    print("   • ColBERTAdapter - Token-level embeddings")
    print("   • ColPaliAdapter - Patch-level visual embeddings")
    
    # Vector stores
    print("\n💾 Vector Stores:")
    print("   • QdrantAdapter - With quantization support")
    
    # Graph stores
    print("\n🔗 Graph Stores:")
    print("   • Neo4jAdapter - Knowledge graph storage")
    
    # Document loaders
    print("\n📄 Document Loaders:")
    print("   • TextLoader - .txt, .md files")
    print("   • PdfLoader - PDF text extraction")
    
    # Chunking
    print("\n✂️  Chunking Strategies:")
    print("   • FixedSizeChunker - Character-based with overlap")
    
    # Retrievers
    print("\n🔍 Retrievers:")
    print("   • VectorRetriever - Similarity search")
    print("   • GraphRetriever - Subgraph traversal")
    print("   • HybridRetriever - Weighted combination")
    print("   • ColBERTRetriever - Late interaction")
    print("   • ColPaliRetriever - Visual document search")
    
    print_header("Example: Building a RAG Pipeline")
    
    print("""
from polyrag.interface import PipelineBuilder, AdapterFactory

pipeline = (
    PipelineBuilder()
    .with_llm(AdapterFactory.create_llm("ollama", model="llama3.2"))
    .with_embedding(AdapterFactory.create_embedding("fastembed"))
    .with_vector_store(AdapterFactory.create_vector_store("qdrant"))
    .with_document_loader(AdapterFactory.create_document_loader("text"))
    .with_chunker(AdapterFactory.create_chunker("fixed_size"))
    .with_collection_name("my_docs")
    .build()
)

# Ingest documents
pipeline.ingest("./documents/")

# Query with streaming
for chunk in pipeline.query_stream("What is machine learning?"):
    print(chunk, end="", flush=True)
""")
    
    print_header("Quick Start")
    
    print("""
1. Install dependencies:
   pip install fastembed qdrant-client requests

2. Start Qdrant:
   docker run -p 6333:6333 qdrant/qdrant

3. Start Ollama:
   ollama serve
   ollama pull llama3.2

4. Run the simple example:
   python polyrag/examples/simple_rag.py

5. Launch Streamlit demo:
   streamlit run polyrag/examples/streamlit_demo.py
""")
    
    print_header("Project Complete! 🎉")
    print("\nPolyRAG is ready for use. Happy building!")


if __name__ == "__main__":
    main()
