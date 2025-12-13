import os
from typing import List, Optional
import google.generativeai as genai
from polyrag.core.ports.embedding_port import TextEmbeddingPort

class GeminiEmbeddingAdapter(TextEmbeddingPort):
    """Adapter for Google Gemini embeddings."""

    def __init__(self, model_name: str = "models/text-embedding-004", api_key: Optional[str] = None):
        """
        Initialize the Gemini embedding adapter.

        Args:
            model_name: The name of the embedding model to use. 
                        Common models: "models/embedding-001", "models/text-embedding-004"
            api_key: Google Gemini API key.
        """
        self._model_name = model_name
        self._dimension: int | None = None
        
        if api_key:
            genai.configure(api_key=api_key)
        elif "GOOGLE_API_KEY" in os.environ:
            genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

    @staticmethod
    def list_models(api_key: Optional[str] = None) -> List[str]:
        """
        List available Gemini models that support embeddings.
        """
        if api_key:
            genai.configure(api_key=api_key)
        elif "GOOGLE_API_KEY" in os.environ:
            genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
            
        try:
            models = []
            for m in genai.list_models():
                if 'embedContent' in m.supported_generation_methods:
                    models.append(m.name.replace('models/', ''))
            return sorted(models)
        except Exception:
            return []

    def embed_text(self, text: str) -> List[float]:
        """
        Embeds a single text string.

        Args:
            text: The text to embed.

        Returns:
            The embedding vector as a list of floats.
        """
        result = genai.embed_content(
            model=self._model_name,
            content=text,
            task_type="retrieval_document" # or retrieval_query depending on usage, but document is generally safe for storage
        )
        embedding = result['embedding']
        if self._dimension is None:
            self._dimension = len(embedding)
        return embedding

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Embeds a list of text strings.

        Args:
            texts: The texts to embed.

        Returns:
            A list of embedding vectors.
        """
        # Batch embedding if supported by API, else loop
        # genai.embed_content supports "content" as a list of strings
        result = genai.embed_content(
            model=self._model_name,
            content=texts,
            task_type="retrieval_document"
        )
        # Result "embedding" key will be a list of lists if input was list
        embeddings = result['embedding']
        
        if self._dimension is None and embeddings:
            self._dimension = len(embeddings[0])
            
        return embeddings

    @property
    def dimension(self) -> int:
        """Returns the dimension of the embedding vector."""
        if self._dimension is None:
            # Compute dimension by embedding a dummy text
            dummy = self.embed_text("dummy")
            self._dimension = len(dummy)
        return self._dimension
