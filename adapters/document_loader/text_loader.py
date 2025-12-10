import os
from typing import List

from polyrag.core.ports.document_loader_port import DocumentLoaderPort
from polyrag.core.models.models import Document, Element, ElementType


class TextLoader(DocumentLoaderPort):
    """Adapter for loading text files (.txt, .md)."""

    def __init__(self, encoding: str = "utf-8"):
        """
        Initialize the TextLoader.

        Args:
            encoding: File encoding to use when reading files.
        """
        self._encoding = encoding

    def load(self, file_path: str) -> Document:
        """
        Loads a text file and returns a Document object.

        Args:
            file_path: Path to the file to load.

        Returns:
            A Document containing the file content as a TextElement.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the file extension is not supported.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        ext = os.path.splitext(file_path)[1].lower()
        if ext not in self.supported_extensions:
            raise ValueError(f"Unsupported file extension: {ext}")

        with open(file_path, "r", encoding=self._encoding) as f:
            content = f.read()

        element = Element(
            content=content,
            type=ElementType.TEXT,
            metadata={"source_file": file_path}
        )

        return Document(
            elements=[element],
            metadata={
                "source_file": file_path,
                "file_name": os.path.basename(file_path),
                "file_extension": ext
            }
        )

    @property
    def supported_extensions(self) -> List[str]:
        """Returns a list of supported file extensions."""
        return [".txt", ".md"]
