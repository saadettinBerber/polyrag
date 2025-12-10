from abc import ABC, abstractmethod
from typing import List, Any
from polyrag.core.models.models import RetrievalResult

class RetrieverPort(ABC):
    """Abstract interface for Retrievers."""

    @abstractmethod
    def retrieve(self, query: str, limit: int = 5, **kwargs) -> List[RetrievalResult]:
        """Retrieves relevant chunks for a given query."""
        pass
