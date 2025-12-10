from abc import ABC, abstractmethod
from typing import List, Any
from polyrag.core.models.models import Document

class DocumentLoaderPort(ABC):
    """Abstract interface for Document Loaders."""

    @abstractmethod
    def load(self, file_path: str) -> Document:
        """Loads a file and returns a Document object."""
        pass

    @property
    @abstractmethod
    def supported_extensions(self) -> List[str]:
        """Returns a list of supported file extensions (e.g., ['.txt', '.md'])."""
        pass
