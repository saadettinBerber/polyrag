from typing import List, Any
import numpy as np
from PIL import Image

try:
    from colpali_engine.models import ColPali
    from colpali_engine.utils.processing_utils import BaseVisualRetrieverProcessor
except ImportError:
    ColPali = None
    BaseVisualRetrieverProcessor = None

from polyrag.core.ports.embedding_port import ImageEmbeddingPort


class ColPaliAdapter(ImageEmbeddingPort):
    """
    Adapter for ColPali patch-level image embeddings.
    
    ColPali uses vision-language models to embed document images
    at the patch level for precise visual document retrieval.
    """

    def __init__(
        self,
        model_name: str = "vidore/colpali-v1.2",
        device: str = "cpu"
    ):
        """
        Initialize the ColPali adapter.

        Args:
            model_name: ColPali model name or path.
            device: Device to run on (cpu/cuda).

        Raises:
            ImportError: If colpali-engine is not installed.
        """
        if ColPali is None:
            raise ImportError(
                "colpali-engine is required. Install with: pip install colpali-engine"
            )
        
        self._model_name = model_name
        self._device = device
        
        # Load model and processor
        self._model = ColPali.from_pretrained(
            model_name,
            torch_dtype="auto"
        ).to(device).eval()
        
        self._processor = BaseVisualRetrieverProcessor.from_pretrained(model_name)
        self._dimension: int = 128  # ColPali default dimension

    def embed_image(self, image: Any) -> List[float]:
        """
        Embeds a single image.
        
        Note: ColPali produces patch-level embeddings. This method
        returns a pooled representation for compatibility.

        Args:
            image: File path or PIL Image.

        Returns:
            Pooled embedding vector.
        """
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        
        # Process image
        batch = self._processor.process_images([image])
        batch = {k: v.to(self._device) for k, v in batch.items()}
        
        # Get embeddings
        with self._model.no_grad():
            outputs = self._model(**batch)
        
        # Pool patch embeddings
        pooled = outputs[0].mean(dim=0).cpu().numpy()
        return pooled.tolist()

    def embed_images(self, images: List[Any]) -> List[List[float]]:
        """
        Embeds multiple images.

        Args:
            images: List of file paths or PIL Images.

        Returns:
            List of pooled embedding vectors.
        """
        pil_images = []
        for img in images:
            if isinstance(img, str):
                pil_images.append(Image.open(img).convert("RGB"))
            else:
                pil_images.append(img.convert("RGB"))
        
        # Process batch
        batch = self._processor.process_images(pil_images)
        batch = {k: v.to(self._device) for k, v in batch.items()}
        
        # Get embeddings
        with self._model.no_grad():
            outputs = self._model(**batch)
        
        results = []
        for emb in outputs:
            pooled = emb.mean(dim=0).cpu().numpy()
            results.append(pooled.tolist())
        
        return results

    def embed_image_patches(self, image: Any) -> np.ndarray:
        """
        Get patch-level embeddings for an image (for late interaction).

        Args:
            image: File path or PIL Image.

        Returns:
            Patch-level embeddings as numpy array.
        """
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        
        batch = self._processor.process_images([image])
        batch = {k: v.to(self._device) for k, v in batch.items()}
        
        with self._model.no_grad():
            outputs = self._model(**batch)
        
        return outputs[0].cpu().numpy()

    @property
    def dimension(self) -> int:
        """Returns the dimension of the embedding vector."""
        return self._dimension
