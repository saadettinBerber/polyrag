from typing import List, Union
import numpy as np

try:
    from colbert import Searcher
    from colbert.infra import ColBERTConfig
    from colbert.modeling.checkpoint import Checkpoint
except ImportError:
    Searcher = None
    ColBERTConfig = None
    Checkpoint = None

from polyrag.core.ports.embedding_port import TextEmbeddingPort


class ColBERTAdapter(TextEmbeddingPort):
    """
    Adapter for ColBERT token-level embeddings.
    
    ColBERT uses late interaction - embedding each token separately
    and computing similarity at query time for better precision.
    """

    def __init__(
        self,
        model_name: str = "colbert-ir/colbertv2.0",
        device: str = "cpu"
    ):
        """
        Initialize the ColBERT adapter.

        Args:
            model_name: ColBERT model name or path.
            device: Device to run on (cpu/cuda).

        Raises:
            ImportError: If colbert is not installed.
        """
        if Checkpoint is None:
            raise ImportError(
                "colbert-ai is required. Install with: pip install colbert-ai"
            )
        
        self._model_name = model_name
        self._device = device
        
        # Initialize checkpoint for encoding
        self._config = ColBERTConfig(
            checkpoint=model_name,
            doc_maxlen=256,
            query_maxlen=32
        )
        self._checkpoint = Checkpoint(model_name, colbert_config=self._config)
        self._dimension: int = 128  # ColBERT default dimension

    def embed_text(self, text: str) -> List[float]:
        """
        Embeds a single text string.
        
        Note: ColBERT produces token-level embeddings. This method
        returns a pooled representation for compatibility.

        Args:
            text: The text to embed.

        Returns:
            Pooled embedding vector.
        """
        embeddings = self._checkpoint.docFromText([text])
        # Pool token embeddings (mean pooling)
        pooled = embeddings[0].mean(dim=0).cpu().numpy()
        return pooled.tolist()

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Embeds multiple text strings.

        Args:
            texts: List of texts to embed.

        Returns:
            List of pooled embedding vectors.
        """
        embeddings = self._checkpoint.docFromText(texts)
        results = []
        for emb in embeddings:
            pooled = emb.mean(dim=0).cpu().numpy()
            results.append(pooled.tolist())
        return results

    def embed_query_tokens(self, query: str) -> np.ndarray:
        """
        Get token-level embeddings for a query (for late interaction).

        Args:
            query: The query text.

        Returns:
            Token-level embeddings as numpy array.
        """
        embeddings = self._checkpoint.queryFromText([query])
        return embeddings[0].cpu().numpy()

    def embed_doc_tokens(self, doc: str) -> np.ndarray:
        """
        Get token-level embeddings for a document (for late interaction).

        Args:
            doc: The document text.

        Returns:
            Token-level embeddings as numpy array.
        """
        embeddings = self._checkpoint.docFromText([doc])
        return embeddings[0].cpu().numpy()

    @property
    def dimension(self) -> int:
        """Returns the dimension of the embedding vector."""
        return self._dimension
