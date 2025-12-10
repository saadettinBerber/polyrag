from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from polyrag.interface.pipeline import PolyRAGPipeline

from polyrag.core.ports.llm_port import LLMPort
from polyrag.core.ports.embedding_port import TextEmbeddingPort
from polyrag.core.ports.vector_store_port import VectorStorePort
from polyrag.core.ports.document_loader_port import DocumentLoaderPort
from polyrag.core.ports.chunking_port import ChunkingPort
from polyrag.core.ports.retriever_port import RetrieverPort


class PipelineBuilder:
    """Fluent API builder for constructing PolyRAG pipelines."""

    def __init__(self):
        """Initialize an empty pipeline builder."""
        self._llm: Optional[LLMPort] = None
        self._embedding: Optional[TextEmbeddingPort] = None
        self._vector_store: Optional[VectorStorePort] = None
        self._document_loader: Optional[DocumentLoaderPort] = None
        self._chunker: Optional[ChunkingPort] = None
        self._retriever: Optional[RetrieverPort] = None
        self._collection_name: str = "polyrag_default"

    def with_llm(self, llm: LLMPort) -> "PipelineBuilder":
        """
        Set the LLM adapter.

        Args:
            llm: An instance implementing LLMPort.

        Returns:
            Self for method chaining.
        """
        self._llm = llm
        return self

    def with_embedding(self, embedding: TextEmbeddingPort) -> "PipelineBuilder":
        """
        Set the embedding adapter.

        Args:
            embedding: An instance implementing TextEmbeddingPort.

        Returns:
            Self for method chaining.
        """
        self._embedding = embedding
        return self

    def with_vector_store(self, vector_store: VectorStorePort) -> "PipelineBuilder":
        """
        Set the vector store adapter.

        Args:
            vector_store: An instance implementing VectorStorePort.

        Returns:
            Self for method chaining.
        """
        self._vector_store = vector_store
        return self

    def with_document_loader(self, loader: DocumentLoaderPort) -> "PipelineBuilder":
        """
        Set the document loader adapter.

        Args:
            loader: An instance implementing DocumentLoaderPort.

        Returns:
            Self for method chaining.
        """
        self._document_loader = loader
        return self

    def with_chunker(self, chunker: ChunkingPort) -> "PipelineBuilder":
        """
        Set the chunking adapter.

        Args:
            chunker: An instance implementing ChunkingPort.

        Returns:
            Self for method chaining.
        """
        self._chunker = chunker
        return self

    def with_retriever(self, retriever: RetrieverPort) -> "PipelineBuilder":
        """
        Set the retriever adapter.

        Args:
            retriever: An instance implementing RetrieverPort.

        Returns:
            Self for method chaining.
        """
        self._retriever = retriever
        return self

    def with_collection_name(self, name: str) -> "PipelineBuilder":
        """
        Set the collection name for vector storage.

        Args:
            name: The collection name.

        Returns:
            Self for method chaining.
        """
        self._collection_name = name
        return self

    def build(self) -> "PolyRAGPipeline":
        """
        Build and return the configured pipeline.

        Returns:
            A configured PolyRAGPipeline instance.

        Raises:
            ValueError: If required components are missing.
        """
        # Validate required components
        if self._llm is None:
            raise ValueError("LLM adapter is required. Use with_llm().")
        if self._embedding is None:
            raise ValueError("Embedding adapter is required. Use with_embedding().")
        if self._vector_store is None:
            raise ValueError("Vector store adapter is required. Use with_vector_store().")
        if self._document_loader is None:
            raise ValueError("Document loader is required. Use with_document_loader().")
        if self._chunker is None:
            raise ValueError("Chunker is required. Use with_chunker().")

        # Import here to avoid circular imports
        from polyrag.interface.pipeline import PolyRAGPipeline
        from polyrag.adapters.retriever.vector_retriever import VectorRetriever

        # Create default retriever if not provided
        retriever = self._retriever
        if retriever is None:
            retriever = VectorRetriever(
                embedding_adapter=self._embedding,
                vector_store_adapter=self._vector_store,
                collection_name=self._collection_name
            )

        return PolyRAGPipeline(
            llm=self._llm,
            embedding=self._embedding,
            vector_store=self._vector_store,
            document_loader=self._document_loader,
            chunker=self._chunker,
            retriever=retriever,
            collection_name=self._collection_name
        )
