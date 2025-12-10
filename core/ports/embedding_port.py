from abc import ABC, abstractmethod
from typing import List, Union, Any

class TextEmbeddingPort(ABC):
    """Abstract interface for text embedding models."""

    @abstractmethod
    def embed_text(self, text: str) -> List[float]:
        """Embeds a single text string."""
        pass

    @abstractmethod
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Embeds a list of text strings."""
        pass

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Returns the dimension of the embedding vector."""
        pass

class ImageEmbeddingPort(ABC):
    """Abstract interface for image embedding models."""

    @abstractmethod
    def embed_image(self, image: Any) -> List[float]:
        """Embeds a single image. Image type depends on implementation (path, bytes, PIL)."""
        pass

    @abstractmethod
    def embed_images(self, images: List[Any]) -> List[List[float]]:
        """Embeds a list of images."""
        pass
    
    @property
    @abstractmethod
    def dimension(self) -> int:
        """Returns the dimension of the embedding vector."""
        pass
