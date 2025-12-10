from typing import List
import numpy as np
from PIL import Image

from polyrag.core.ports.retriever_port import RetrieverPort
from polyrag.core.ports.vector_store_port import VectorStorePort
from polyrag.core.models.models import RetrievalResult, Chunk, ChunkType
from polyrag.adapters.embedding.colpali_adapter import ColPaliAdapter


class ColPaliRetriever(RetrieverPort):
    """
    Retriever using ColPali for vision-driven document retrieval.
    
    Uses patch-level embeddings for visual document understanding,
    ideal for documents where layout and visual elements matter.
    """

    def __init__(
        self,
        colpali_adapter: ColPaliAdapter,
        vector_store: VectorStorePort,
        collection_name: str
    ):
        """
        Initialize the ColPali retriever.

        Args:
            colpali_adapter: ColPali embedding adapter.
            vector_store: Vector store for retrieval.
            collection_name: Collection name.
        """
        self._colpali = colpali_adapter
        self._vector_store = vector_store
        self._collection_name = collection_name

    def retrieve(self, query: str, limit: int = 5, **kwargs) -> List[RetrievalResult]:
        """
        Retrieves relevant document images using text query.

        Note: ColPali embeds document *images*, so this retriever
        is meant for visual document search scenarios.

        Args:
            query: The text search query.
            limit: Maximum number of results.
            **kwargs: Additional parameters.

        Returns:
            List of RetrievalResult objects.
        """
        # For text queries, we use the processor's text encoding
        # ColPali can match text queries to document image patches
        
        # Get pooled query embedding (using image of rendered text or direct encoding)
        # For simplicity, we create a simple text image
        query_image = self._create_text_image(query)
        query_vector = self._colpali.embed_image(query_image)
        
        # Search vector store
        results = self._vector_store.search(
            collection_name=self._collection_name,
            query_vector=query_vector,
            limit=limit
        )
        
        # Update source
        updated_results = []
        for result in results:
            updated_results.append(RetrievalResult(
                chunk=result.chunk,
                score=result.score,
                source="colpali",
                metadata=result.metadata
            ))
        
        return updated_results

    def retrieve_by_image(self, query_image: Image.Image, limit: int = 5) -> List[RetrievalResult]:
        """
        Retrieves similar document images using an image query.

        Args:
            query_image: Query image.
            limit: Maximum number of results.

        Returns:
            List of RetrievalResult objects.
        """
        query_vector = self._colpali.embed_image(query_image)
        
        results = self._vector_store.search(
            collection_name=self._collection_name,
            query_vector=query_vector,
            limit=limit
        )
        
        return [
            RetrievalResult(
                chunk=r.chunk,
                score=r.score,
                source="colpali",
                metadata=r.metadata
            )
            for r in results
        ]

    def _create_text_image(self, text: str, size: tuple = (224, 224)) -> Image.Image:
        """
        Create a simple image with text for query encoding.

        Args:
            text: Text to render.
            size: Image size.

        Returns:
            PIL Image with rendered text.
        """
        from PIL import ImageDraw, ImageFont
        
        img = Image.new("RGB", size, color="white")
        draw = ImageDraw.Draw(img)
        
        try:
            font = ImageFont.truetype("arial.ttf", 16)
        except:
            font = ImageFont.load_default()
        
        # Center text
        draw.text((10, size[1] // 2 - 10), text[:50], fill="black", font=font)
        
        return img
