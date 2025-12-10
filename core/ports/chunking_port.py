from abc import ABC, abstractmethod
from typing import List
from polyrag.core.models.models import Document, Chunk

class ChunkingPort(ABC):
    """Abstract interface for Chunking strategies."""

    @abstractmethod
    def chunk(self, document: Document) -> List[Chunk]:
        """Splits a Document into a list of Chunks."""
        pass
