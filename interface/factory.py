"""
Factory module for creating adapters from configuration strings.
"""
from typing import Optional

from polyrag.core.ports.llm_port import LLMPort
from polyrag.core.ports.embedding_port import TextEmbeddingPort
from polyrag.core.ports.vector_store_port import VectorStorePort
from polyrag.core.ports.document_loader_port import DocumentLoaderPort
from polyrag.core.ports.chunking_port import ChunkingPort


class AdapterFactory:
    """Factory for creating adapters from configuration."""

    @staticmethod
    def create_llm(provider: str = "ollama", **kwargs) -> LLMPort:
        """
        Create an LLM adapter.

        Args:
            provider: LLM provider name (ollama, openai, claude, gemini).
            **kwargs: Provider-specific arguments.

        Returns:
            An LLM adapter instance.
        """
        if provider.lower() == "ollama":
            from polyrag.adapters.llm.ollama_adapter import OllamaAdapter
            return OllamaAdapter(
                model=kwargs.get("model", "llama3.2"),
                base_url=kwargs.get("base_url", "http://localhost:11434")
            )
        else:
            raise ValueError(f"Unknown LLM provider: {provider}")

    @staticmethod
    def create_embedding(provider: str = "fastembed", **kwargs) -> TextEmbeddingPort:
        """
        Create an embedding adapter.

        Args:
            provider: Embedding provider name.
            **kwargs: Provider-specific arguments.

        Returns:
            An embedding adapter instance.
        """
        if provider.lower() == "fastembed":
            from polyrag.adapters.embedding.fastembed_adapter import FastEmbedAdapter
            return FastEmbedAdapter(
                model_name=kwargs.get("model_name", "BAAI/bge-small-en-v1.5")
            )
        elif provider.lower() == "clip":
            from polyrag.adapters.embedding.clip_adapter import CLIPAdapter
            return CLIPAdapter(
                model_name=kwargs.get("model_name", "clip-ViT-B-32")
            )
        else:
            raise ValueError(f"Unknown embedding provider: {provider}")

    @staticmethod
    def create_vector_store(provider: str = "qdrant", **kwargs) -> VectorStorePort:
        """
        Create a vector store adapter.

        Args:
            provider: Vector store provider name.
            **kwargs: Provider-specific arguments.

        Returns:
            A vector store adapter instance.
        """
        if provider.lower() == "qdrant":
            from polyrag.adapters.vector_store.qdrant_adapter import QdrantAdapter
            return QdrantAdapter(
                host=kwargs.get("host", "localhost"),
                port=kwargs.get("port", 6333),
                url=kwargs.get("url"),
                api_key=kwargs.get("api_key")
            )
        else:
            raise ValueError(f"Unknown vector store provider: {provider}")

    @staticmethod
    def create_document_loader(loader_type: str = "text", **kwargs) -> DocumentLoaderPort:
        """
        Create a document loader adapter.

        Args:
            loader_type: Type of loader (text, pdf, image).
            **kwargs: Loader-specific arguments.

        Returns:
            A document loader adapter instance.
        """
        if loader_type.lower() == "text":
            from polyrag.adapters.document_loader.text_loader import TextLoader
            return TextLoader(encoding=kwargs.get("encoding", "utf-8"))
        elif loader_type.lower() == "pdf":
            from polyrag.adapters.document_loader.pdf_loader import PdfLoader
            return PdfLoader()
        else:
            raise ValueError(f"Unknown loader type: {loader_type}")

    @staticmethod
    def create_chunker(chunker_type: str = "fixed_size", **kwargs) -> ChunkingPort:
        """
        Create a chunking adapter.

        Args:
            chunker_type: Type of chunker.
            **kwargs: Chunker-specific arguments.

        Returns:
            A chunking adapter instance.
        """
        if chunker_type.lower() == "fixed_size":
            from polyrag.adapters.chunking.fixed_size_chunker import FixedSizeChunker
            return FixedSizeChunker(
                chunk_size=kwargs.get("chunk_size", 500),
                chunk_overlap=kwargs.get("chunk_overlap", 50)
            )
        else:
            raise ValueError(f"Unknown chunker type: {chunker_type}")
