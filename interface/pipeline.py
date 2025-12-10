import os
from typing import List, Iterator, Optional

from polyrag.core.ports.llm_port import LLMPort
from polyrag.core.ports.embedding_port import TextEmbeddingPort
from polyrag.core.ports.vector_store_port import VectorStorePort
from polyrag.core.ports.document_loader_port import DocumentLoaderPort
from polyrag.core.ports.chunking_port import ChunkingPort
from polyrag.core.ports.retriever_port import RetrieverPort
from polyrag.core.models.models import Document, Chunk, RetrievalResult


class PolyRAGPipeline:
    """Main orchestrator for RAG operations."""

    DEFAULT_SYSTEM_PROMPT = """You are a helpful assistant. Answer the user's question based on the provided context.
If the context doesn't contain relevant information, say so honestly.

Context:
{context}
"""

    def __init__(
        self,
        llm: LLMPort,
        embedding: TextEmbeddingPort,
        vector_store: VectorStorePort,
        document_loader: DocumentLoaderPort,
        chunker: ChunkingPort,
        retriever: RetrieverPort,
        collection_name: str = "polyrag_default"
    ):
        """
        Initialize the RAG pipeline.

        Args:
            llm: LLM adapter for text generation.
            embedding: Embedding adapter for vectorization.
            vector_store: Vector store adapter for storage/search.
            document_loader: Document loader for file ingestion.
            chunker: Chunking adapter for splitting documents.
            retriever: Retriever adapter for similarity search.
            collection_name: Name of the vector collection.
        """
        self._llm = llm
        self._embedding = embedding
        self._vector_store = vector_store
        self._document_loader = document_loader
        self._chunker = chunker
        self._retriever = retriever
        self._collection_name = collection_name
        self._collection_initialized = False

    def _ensure_collection(self):
        """Lazily initialize the vector collection."""
        if not self._collection_initialized:
            self._vector_store.create_collection(
                collection_name=self._collection_name,
                dimension=self._embedding.dimension
            )
            self._collection_initialized = True

    def ingest(self, path: str) -> int:
        """
        Ingest a file or directory into the vector store.

        Args:
            path: Path to a file or directory.

        Returns:
            Number of chunks ingested.
        """
        self._ensure_collection()

        documents: List[Document] = []

        if os.path.isfile(path):
            doc = self._document_loader.load(path)
            documents.append(doc)
        elif os.path.isdir(path):
            for root, _, files in os.walk(path):
                for file in files:
                    file_path = os.path.join(root, file)
                    ext = os.path.splitext(file)[1].lower()
                    if ext in self._document_loader.supported_extensions:
                        try:
                            doc = self._document_loader.load(file_path)
                            documents.append(doc)
                        except Exception as e:
                            print(f"Warning: Could not load {file_path}: {e}")
        else:
            raise FileNotFoundError(f"Path not found: {path}")

        # Chunk all documents
        all_chunks: List[Chunk] = []
        for doc in documents:
            chunks = self._chunker.chunk(doc)
            all_chunks.extend(chunks)

        # Embed all chunks
        if all_chunks:
            texts = [c.content for c in all_chunks]
            embeddings = self._embedding.embed_texts(texts)
            for chunk, embedding in zip(all_chunks, embeddings):
                chunk.embedding = embedding

            # Store in vector DB
            self._vector_store.insert(self._collection_name, all_chunks)

        return len(all_chunks)

    def query(
        self,
        question: str,
        top_k: int = 5,
        system_prompt: Optional[str] = None
    ) -> str:
        """
        Query the RAG pipeline.

        Args:
            question: The user's question.
            top_k: Number of chunks to retrieve.
            system_prompt: Optional custom system prompt (use {context} placeholder).

        Returns:
            The generated answer.
        """
        # Retrieve relevant chunks
        results = self._retriever.retrieve(question, limit=top_k)

        # Build context from retrieved chunks
        context = self._build_context(results)

        # Use default or custom system prompt
        prompt_template = system_prompt or self.DEFAULT_SYSTEM_PROMPT
        formatted_system = prompt_template.format(context=context)

        # Generate response
        response = self._llm.generate(question, system_prompt=formatted_system)
        return response

    def query_stream(
        self,
        question: str,
        top_k: int = 5,
        system_prompt: Optional[str] = None
    ) -> Iterator[str]:
        """
        Query the RAG pipeline with streaming response.

        Args:
            question: The user's question.
            top_k: Number of chunks to retrieve.
            system_prompt: Optional custom system prompt.

        Yields:
            Chunks of the generated answer.
        """
        # Retrieve relevant chunks
        results = self._retriever.retrieve(question, limit=top_k)

        # Build context
        context = self._build_context(results)

        # Format system prompt
        prompt_template = system_prompt or self.DEFAULT_SYSTEM_PROMPT
        formatted_system = prompt_template.format(context=context)

        # Stream response
        for chunk in self._llm.generate_stream(question, system_prompt=formatted_system):
            yield chunk

    def _build_context(self, results: List[RetrievalResult]) -> str:
        """
        Build context string from retrieval results.

        Args:
            results: List of retrieval results.

        Returns:
            Formatted context string.
        """
        if not results:
            return "No relevant context found."

        context_parts = []
        for i, result in enumerate(results, 1):
            source = result.chunk.metadata.get("file_name", "Unknown source")
            context_parts.append(f"[{i}] (Source: {source}, Score: {result.score:.3f})\n{result.chunk.content}")

        return "\n\n".join(context_parts)

    def get_retrieval_results(self, question: str, top_k: int = 5) -> List[RetrievalResult]:
        """
        Get raw retrieval results without LLM generation.

        Args:
            question: The search query.
            top_k: Number of results.

        Returns:
            List of RetrievalResult objects.
        """
        return self._retriever.retrieve(question, limit=top_k)
