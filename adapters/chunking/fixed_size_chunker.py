from typing import List

from polyrag.core.ports.chunking_port import ChunkingPort
from polyrag.core.models.models import Document, Chunk, ChunkType, ElementType


class FixedSizeChunker(ChunkingPort):
    """Adapter for fixed-size character-based chunking."""

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        """
        Initialize the FixedSizeChunker.

        Args:
            chunk_size: Maximum number of characters per chunk.
            chunk_overlap: Number of overlapping characters between chunks.
        """
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")
        
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap

    def chunk(self, document: Document) -> List[Chunk]:
        """
        Splits a Document into a list of Chunks.

        Args:
            document: The document to split.

        Returns:
            A list of Chunk objects.
        """
        chunks = []
        
        # Process text elements
        for element in document.elements:
            if element.type == ElementType.TEXT:
                text = element.content
                text_chunks = self._split_text(text)
                
                for i, chunk_text in enumerate(text_chunks):
                    chunk = Chunk(
                        content=chunk_text,
                        chunk_type=ChunkType.TEXT,
                        source_document_id=document.id,
                        metadata={
                            "chunk_index": i,
                            "total_chunks": len(text_chunks),
                            **element.metadata,
                            **document.metadata
                        }
                    )
                    chunks.append(chunk)
        
        return chunks

    def _split_text(self, text: str) -> List[str]:
        """
        Splits text into chunks of fixed size with overlap.

        Args:
            text: The text to split.

        Returns:
            A list of text chunks.
        """
        if not text:
            return []

        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self._chunk_size
            chunk = text[start:end]
            chunks.append(chunk)
            
            # Move start position, accounting for overlap
            start = end - self._chunk_overlap
            
            # Avoid infinite loop if overlap equals or exceeds remaining text
            if start >= len(text):
                break

        return chunks
