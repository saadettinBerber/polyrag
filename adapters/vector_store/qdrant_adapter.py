from typing import List, Dict, Any, Optional

from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams,
    Distance,
    PointStruct,
    Filter,
    BinaryQuantization,
    BinaryQuantizationConfig,
    ScalarQuantization,
    ScalarQuantizationConfig,
)

from polyrag.core.ports.vector_store_port import VectorStorePort
from polyrag.core.models.models import Chunk, RetrievalResult, QuantizationConfig, QuantizationType, ChunkType


class QdrantAdapter(VectorStorePort):
    """Adapter for Qdrant vector database."""

    def __init__(self, host: str = "localhost", port: int = 6333, url: str = None, api_key: str = None):
        """
        Initialize the Qdrant adapter.

        Args:
            host: Qdrant server host.
            port: Qdrant server port.
            url: Optional URL for cloud deployment.
            api_key: Optional API key for cloud deployment.
        """
        if url:
            self._client = QdrantClient(url=url, api_key=api_key)
        else:
            self._client = QdrantClient(host=host, port=port)

    def create_collection(
        self,
        collection_name: str,
        dimension: int,
        quantization_config: Optional[QuantizationConfig] = None
    ):
        """
        Creates a new collection.

        Args:
            collection_name: Name of the collection.
            dimension: Dimension of the vectors.
            quantization_config: Optional quantization settings.
        """
        qdrant_quantization = None
        if quantization_config and quantization_config.type == QuantizationType.BINARY:
            qdrant_quantization = BinaryQuantization(
                binary=BinaryQuantizationConfig(
                    always_ram=quantization_config.always_ram
                )
            )

        self._client.recreate_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=dimension,
                distance=Distance.COSINE
            ),
            quantization_config=qdrant_quantization
        )

    def insert(self, collection_name: str, chunks: List[Chunk]):
        """
        Inserts chunks into the collection.

        Args:
            collection_name: Name of the collection.
            chunks: List of chunks to insert.
        """
        points = []
        for chunk in chunks:
            if chunk.embedding is None:
                raise ValueError(f"Chunk {chunk.id} has no embedding")
            
            payload = {
                "content": chunk.content,
                "chunk_type": chunk.chunk_type.value,
                "source_document_id": chunk.source_document_id,
                **chunk.metadata
            }
            
            points.append(PointStruct(
                id=chunk.id,
                vector=chunk.embedding,
                payload=payload
            ))

        self._client.upsert(collection_name=collection_name, points=points)

    def search(
        self,
        collection_name: str,
        query_vector: List[float],
        limit: int = 5,
        filter: Optional[Dict] = None
    ) -> List[RetrievalResult]:
        """
        Searches for similar chunks using a query vector.

        Args:
            collection_name: Name of the collection.
            query_vector: The query embedding vector.
            limit: Maximum number of results to return.
            filter: Optional filter conditions.

        Returns:
            List of RetrievalResult objects.
        """
        qdrant_filter = None
        if filter:
            qdrant_filter = Filter(**filter)

        results = self._client.search(
            collection_name=collection_name,
            query_vector=query_vector,
            limit=limit,
            query_filter=qdrant_filter
        )

        retrieval_results = []
        for hit in results:
            payload = hit.payload or {}
            chunk = Chunk(
                content=payload.get("content", ""),
                chunk_type=ChunkType(payload.get("chunk_type", "text")),
                source_document_id=payload.get("source_document_id", ""),
                id=str(hit.id),
                metadata={k: v for k, v in payload.items() 
                          if k not in ["content", "chunk_type", "source_document_id"]}
            )
            retrieval_results.append(RetrievalResult(
                chunk=chunk,
                score=hit.score,
                source="vector",
                metadata=payload
            ))

        return retrieval_results

    def delete_collection(self, collection_name: str):
        """
        Deletes a collection.

        Args:
            collection_name: Name of the collection to delete.
        """
        self._client.delete_collection(collection_name=collection_name)
