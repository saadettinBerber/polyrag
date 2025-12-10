"""
GraphRAG Example

This example demonstrates knowledge graph-based RAG:
1. Create a knowledge graph from documents
2. Use graph traversal for contextual retrieval
3. Combine with vector search via HybridRetriever

Prerequisites:
- Neo4j running (bolt://localhost:7687)
- Qdrant running (localhost:6333)
"""

from polyrag.adapters.graph_store.neo4j_adapter import Neo4jAdapter
from polyrag.adapters.vector_store.qdrant_adapter import QdrantAdapter
from polyrag.adapters.embedding.fastembed_adapter import FastEmbedAdapter
from polyrag.adapters.retriever.vector_retriever import VectorRetriever
from polyrag.adapters.retriever.graph_retriever import GraphRetriever
from polyrag.adapters.retriever.hybrid_retriever import HybridRetriever
from polyrag.adapters.llm.ollama_adapter import OllamaAdapter


def main():
    print("=== GraphRAG Demo ===\n")
    
    # Initialize adapters
    print("Initializing adapters...")
    
    # Graph store
    graph_store = Neo4jAdapter(
        uri="bolt://localhost:7687",
        user="neo4j",
        password="password"
    )
    graph_store.connect()
    
    # Vector store
    vector_store = QdrantAdapter(host="localhost", port=6333)
    embedding = FastEmbedAdapter()
    
    collection_name = "graph_rag_demo"
    vector_store.create_collection(collection_name, dimension=embedding.dimension)
    
    # LLM
    llm = OllamaAdapter(model="llama3.2")
    
    # Create sample knowledge graph
    print("\nBuilding knowledge graph...")
    
    # Add document nodes
    doc1_id = graph_store.add_node("Document", {
        "id": "doc1",
        "title": "Machine Learning Basics",
        "content": "Machine learning is a subset of AI that enables systems to learn from data."
    })
    
    doc2_id = graph_store.add_node("Document", {
        "id": "doc2", 
        "title": "Deep Learning",
        "content": "Deep learning uses neural networks with multiple layers to process data."
    })
    
    # Add entity nodes
    entity1_id = graph_store.add_node("Entity", {
        "id": "entity1",
        "name": "Neural Networks",
        "content": "Neural networks are computing systems inspired by biological neural networks."
    })
    
    entity2_id = graph_store.add_node("Entity", {
        "id": "entity2",
        "name": "AI",
        "content": "Artificial Intelligence is the simulation of human intelligence by machines."
    })
    
    # Add relationships
    graph_store.add_edge("doc1", "entity2", "MENTIONS")
    graph_store.add_edge("doc2", "entity1", "EXPLAINS")
    graph_store.add_edge("entity1", "entity2", "PART_OF")
    
    print("Knowledge graph created with 4 nodes and 3 relationships.")
    
    # Create retrievers
    graph_retriever = GraphRetriever(
        graph_store=graph_store,
        node_label="Document",
        depth=2
    )
    
    vector_retriever = VectorRetriever(
        embedding_adapter=embedding,
        vector_store_adapter=vector_store,
        collection_name=collection_name
    )
    
    # Hybrid retriever with 70% vector, 30% graph
    hybrid_retriever = HybridRetriever(
        retrievers=[vector_retriever, graph_retriever],
        weights=[0.7, 0.3]
    )
    
    # Query
    print("\n--- Querying with GraphRetriever ---")
    query = "neural networks"
    
    results = graph_retriever.retrieve(query, limit=3)
    for i, result in enumerate(results, 1):
        print(f"{i}. {result.chunk.content[:100]}...")
        print(f"   Subgraph nodes: {result.chunk.metadata.get('subgraph_nodes', 0)}")
    
    # Generate answer (if LLM is available)
    print("\n--- Generating answer ---")
    context = "\n".join([r.chunk.content for r in results])
    
    try:
        system_prompt = f"Answer based on this context:\n{context}"
        answer = llm.generate(query, system_prompt=system_prompt)
        print(f"Q: {query}")
        print(f"A: {answer}")
    except Exception as e:
        print(f"LLM unavailable: {e}")
    
    # Cleanup
    print("\nCleaning up...")
    vector_store.delete_collection(collection_name)
    graph_store.close()
    
    print("Done!")


if __name__ == "__main__":
    main()
