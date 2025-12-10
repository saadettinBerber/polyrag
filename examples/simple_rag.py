"""
Simple RAG Example

This example demonstrates a complete RAG workflow:
1. Load a text file
2. Chunk, embed, and store in vector database
3. Query with natural language

Prerequisites:
- Ollama running locally (http://localhost:11434)
- Qdrant running locally (http://localhost:6333)
"""

from polyrag.interface.builder import PipelineBuilder
from polyrag.interface.factory import AdapterFactory


def main():
    # Create adapters using factory
    llm = AdapterFactory.create_llm("ollama", model="llama3.2")
    embedding = AdapterFactory.create_embedding("fastembed")
    vector_store = AdapterFactory.create_vector_store("qdrant")
    loader = AdapterFactory.create_document_loader("text")
    chunker = AdapterFactory.create_chunker("fixed_size", chunk_size=500, chunk_overlap=50)

    # Build pipeline using fluent API
    pipeline = (
        PipelineBuilder()
        .with_llm(llm)
        .with_embedding(embedding)
        .with_vector_store(vector_store)
        .with_document_loader(loader)
        .with_chunker(chunker)
        .with_collection_name("simple_rag_demo")
        .build()
    )

    # Ingest sample data
    print("Ingesting documents...")
    sample_file = "sample_data.txt"  # Create this file with your content
    
    try:
        num_chunks = pipeline.ingest(sample_file)
        print(f"Ingested {num_chunks} chunks.")
    except FileNotFoundError:
        print(f"Please create a '{sample_file}' file with some content to test.")
        return

    # Query the pipeline
    print("\n--- Querying the RAG pipeline ---")
    question = "What is the main topic of the document?"
    
    print(f"Question: {question}")
    print("\nAnswer:")
    
    # Use streaming response
    for chunk in pipeline.query_stream(question):
        print(chunk, end="", flush=True)
    
    print("\n")


if __name__ == "__main__":
    main()
