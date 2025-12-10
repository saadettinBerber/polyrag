from typing import List, Any, Union
from PIL import Image

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

from polyrag.core.ports.embedding_port import TextEmbeddingPort, ImageEmbeddingPort


class CLIPAdapter(TextEmbeddingPort, ImageEmbeddingPort):
    """
    Adapter for CLIP model using sentence-transformers.
    
    Embeds both text and images into the same vector space,
    enabling cross-modal similarity search.
    """

    def __init__(self, model_name: str = "clip-ViT-B-32"):
        """
        Initialize the CLIP adapter.

        Args:
            model_name: The name of the CLIP model to use.

        Raises:
            ImportError: If sentence-transformers is not installed.
        """
        if SentenceTransformer is None:
            raise ImportError(
                "sentence-transformers is required for CLIPAdapter. "
                "Install with: pip install sentence-transformers"
            )
        
        self._model_name = model_name
        self._model = SentenceTransformer(model_name)
        self._dimension: int | None = None

    def embed_text(self, text: str) -> List[float]:
        """
        Embeds a single text string.

        Args:
            text: The text to embed.

        Returns:
            The embedding vector.
        """
        embedding = self._model.encode(text, convert_to_numpy=True)
        result = embedding.tolist()
        if self._dimension is None:
            self._dimension = len(result)
        return result

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Embeds multiple text strings.

        Args:
            texts: List of texts to embed.

        Returns:
            List of embedding vectors.
        """
        embeddings = self._model.encode(texts, convert_to_numpy=True)
        results = [emb.tolist() for emb in embeddings]
        if self._dimension is None and results:
            self._dimension = len(results[0])
        return results

    def embed_image(self, image: Any) -> List[float]:
        """
        Embeds a single image.

        Args:
            image: Either a file path (str) or PIL Image object.

        Returns:
            The embedding vector.
        """
        if isinstance(image, str):
            image = Image.open(image)
        
        embedding = self._model.encode(image, convert_to_numpy=True)
        result = embedding.tolist()
        if self._dimension is None:
            self._dimension = len(result)
        return result

    def embed_images(self, images: List[Any]) -> List[List[float]]:
        """
        Embeds multiple images.

        Args:
            images: List of file paths or PIL Image objects.

        Returns:
            List of embedding vectors.
        """
        pil_images = []
        for img in images:
            if isinstance(img, str):
                pil_images.append(Image.open(img))
            else:
                pil_images.append(img)
        
        embeddings = self._model.encode(pil_images, convert_to_numpy=True)
        results = [emb.tolist() for emb in embeddings]
        if self._dimension is None and results:
            self._dimension = len(results[0])
        return results

    @property
    def dimension(self) -> int:
        """Returns the dimension of the embedding vector."""
        if self._dimension is None:
            # Compute dimension by embedding a dummy text
            dummy = self._model.encode("dummy", convert_to_numpy=True)
            self._dimension = len(dummy.tolist())
        return self._dimension
