from typing import List

from fastembed import TextEmbedding

from polyrag.core.ports.embedding_port import TextEmbeddingPort


class FastEmbedAdapter(TextEmbeddingPort):
    """Adapter for FastEmbed text embedding library."""

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        """
        Initialize the FastEmbed adapter.

        Args:
            model_name: The name of the embedding model to use.
        """
        self._model_name = model_name
        self._model = TextEmbedding(model_name=model_name)
        self._dimension: int | None = None

    def embed_text(self, text: str) -> List[float]:
        """
        Embeds a single text string.

        Args:
            text: The text to embed.

        Returns:
            The embedding vector as a list of floats.
        """
        embeddings = list(self._model.embed([text]))
        result = embeddings[0].tolist()
        if self._dimension is None:
            self._dimension = len(result)
        return result

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Embeds a list of text strings.

        Args:
            texts: The texts to embed.

        Returns:
            A list of embedding vectors.
        """
        embeddings = list(self._model.embed(texts))
        results = [emb.tolist() for emb in embeddings]
        if self._dimension is None and results:
            self._dimension = len(results[0])
        return results

    @property
    def dimension(self) -> int:
        """Returns the dimension of the embedding vector."""
        if self._dimension is None:
            # Compute dimension by embedding a dummy text
            dummy = list(self._model.embed(["dummy"]))
            self._dimension = len(dummy[0].tolist())
        return self._dimension
