from typing import List

from polyrag.core.ports.retriever_port import RetrieverPort
from polyrag.core.ports.embedding_port import TextEmbeddingPort
from polyrag.core.ports.vector_store_port import VectorStorePort
from polyrag.core.models.models import RetrievalResult


class VectorRetriever(RetrieverPort):
    """Adapter for vector-based retrieval using embedding and vector store."""

    def __init__(
        self,
        embedding_adapter: TextEmbeddingPort,
        vector_store_adapter: VectorStorePort,
        collection_name: str
    ):
        """
        Initialize the VectorRetriever.

        Args:
            embedding_adapter: The embedding adapter to use for query embedding.
            vector_store_adapter: The vector store adapter to search.
            collection_name: The name of the collection to search in.
        """
        self._embedding_adapter = embedding_adapter
        self._vector_store_adapter = vector_store_adapter
        self._collection_name = collection_name

    def retrieve(self, query: str, limit: int = 5, **kwargs) -> List[RetrievalResult]:
        """
        Retrieves relevant chunks for a given query.

        Args:
            query: The search query.
            limit: Maximum number of results to return.
            **kwargs: Additional parameters (e.g., filter).

        Returns:
            A list of RetrievalResult objects.
        """
        # Embed the query
        query_vector = self._embedding_adapter.embed_text(query)

        # Search the vector store
        filter_dict = kwargs.get("filter", None)
        results = self._vector_store_adapter.search(
            collection_name=self._collection_name,
            query_vector=query_vector,
            limit=limit,
            filter=filter_dict
        )

        return results
