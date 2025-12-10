from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from polyrag.core.models.models import Chunk, RetrievalResult, QuantizationConfig

class VectorStorePort(ABC):
    """Abstract interface for Vector Databases."""

    @abstractmethod
    def create_collection(self, collection_name: str, dimension: int, quantization_config: Optional[QuantizationConfig] = None):
        """Creates a new collection."""
        pass

    @abstractmethod
    def insert(self, collection_name: str, chunks: List[Chunk]):
        """Inserts chunks into the collection."""
        pass

    @abstractmethod
    def search(self, collection_name: str, query_vector: List[float], limit: int = 5, filter: Optional[Dict] = None) -> List[RetrievalResult]:
        """Searches for similar chunks using a query vector."""
        pass

    @abstractmethod
    def delete_collection(self, collection_name: str):
        """Deletes a collection."""
        pass
