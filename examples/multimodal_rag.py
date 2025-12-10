"""
Multimodal RAG Example

This example demonstrates multimodal RAG workflow:
1. Embed images using CLIP
2. Store embeddings in vector database
3. Search images using text queries

Prerequisites:
- Qdrant running locally
- sentence-transformers installed
"""

import os
from typing import List

from polyrag.adapters.embedding.clip_adapter import CLIPAdapter
from polyrag.adapters.vector_store.qdrant_adapter import QdrantAdapter
from polyrag.core.models.models import Chunk, ChunkType


def main():
    print("=== Multimodal RAG with CLIP ===\n")
    
    # Initialize adapters
    print("Initializing CLIP embedding adapter...")
    clip = CLIPAdapter(model_name="clip-ViT-B-32")
    
    print("Initializing Qdrant vector store...")
    vector_store = QdrantAdapter(host="localhost", port=6333)
    
    collection_name = "multimodal_demo"
    
    # Create collection
    print(f"Creating collection '{collection_name}'...")
    vector_store.create_collection(
        collection_name=collection_name,
        dimension=clip.dimension
    )
    
    # Example: Index some images
    # In a real scenario, you would load actual image files
    sample_images = [
        "sample_images/cat.jpg",
        "sample_images/dog.jpg",
        "sample_images/car.jpg",
    ]
    
    # Check if sample images exist
    existing_images = [img for img in sample_images if os.path.exists(img)]
    
    if not existing_images:
        print("\nNo sample images found. Creating demo with text descriptions instead.")
        print("In a real scenario, you would have actual image files.\n")
        
        # Demo: Use text descriptions as proxies for images
        image_descriptions = [
            ("A cute orange cat sleeping on a couch", "cat.jpg"),
            ("A golden retriever playing in a park", "dog.jpg"),
            ("A red sports car on a highway", "car.jpg"),
            ("A beautiful sunset over the ocean", "sunset.jpg"),
            ("A mountain landscape with snow", "mountain.jpg"),
        ]
        
        chunks = []
        for description, filename in image_descriptions:
            embedding = clip.embed_text(description)
            chunk = Chunk(
                content=description,
                chunk_type=ChunkType.IMAGE,  # Marked as image type
                source_document_id=filename,
                embedding=embedding,
                metadata={"filename": filename, "description": description}
            )
            chunks.append(chunk)
        
        print(f"Indexing {len(chunks)} image descriptions...")
        vector_store.insert(collection_name, chunks)
    else:
        print(f"Found {len(existing_images)} images. Indexing...")
        embeddings = clip.embed_images(existing_images)
        
        chunks = []
        for img_path, embedding in zip(existing_images, embeddings):
            chunk = Chunk(
                content=img_path,
                chunk_type=ChunkType.IMAGE,
                source_document_id=img_path,
                embedding=embedding,
                metadata={"filename": os.path.basename(img_path)}
            )
            chunks.append(chunk)
        
        vector_store.insert(collection_name, chunks)
    
    # Search with text query
    print("\n--- Searching with text queries ---\n")
    
    queries = [
        "a fluffy pet",
        "vehicle",
        "nature scenery",
    ]
    
    for query in queries:
        print(f"Query: '{query}'")
        query_embedding = clip.embed_text(query)
        results = vector_store.search(
            collection_name=collection_name,
            query_vector=query_embedding,
            limit=2
        )
        
        for i, result in enumerate(results, 1):
            print(f"  {i}. {result.chunk.metadata.get('filename', 'Unknown')} "
                  f"(score: {result.score:.3f})")
        print()
    
    # Cleanup
    print("Cleaning up...")
    vector_store.delete_collection(collection_name)
    print("Done!")


if __name__ == "__main__":
    main()
