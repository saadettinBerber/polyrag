import os
from typing import List

try:
    from pypdf import PdfReader
except ImportError:
    PdfReader = None

from polyrag.core.ports.document_loader_port import DocumentLoaderPort
from polyrag.core.models.models import Document, Element, ElementType


class PdfLoader(DocumentLoaderPort):
    """Adapter for loading PDF files using pypdf."""

    def __init__(self):
        """
        Initialize the PdfLoader.

        Raises:
            ImportError: If pypdf is not installed.
        """
        if PdfReader is None:
            raise ImportError("pypdf is required for PdfLoader. Install with: pip install pypdf")

    def load(self, file_path: str) -> Document:
        """
        Loads a PDF file and extracts text content.

        Args:
            file_path: Path to the PDF file.

        Returns:
            A Document containing the extracted text as TextElements.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the file extension is not supported.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        ext = os.path.splitext(file_path)[1].lower()
        if ext not in self.supported_extensions:
            raise ValueError(f"Unsupported file extension: {ext}")

        reader = PdfReader(file_path)
        elements = []

        for page_num, page in enumerate(reader.pages):
            text = page.extract_text()
            if text and text.strip():
                element = Element(
                    content=text,
                    type=ElementType.TEXT,
                    metadata={
                        "source_file": file_path,
                        "page_number": page_num + 1,
                        "total_pages": len(reader.pages)
                    }
                )
                elements.append(element)

        return Document(
            elements=elements,
            metadata={
                "source_file": file_path,
                "file_name": os.path.basename(file_path),
                "file_extension": ext,
                "total_pages": len(reader.pages)
            }
        )

    @property
    def supported_extensions(self) -> List[str]:
        """Returns a list of supported file extensions."""
        return [".pdf"]
