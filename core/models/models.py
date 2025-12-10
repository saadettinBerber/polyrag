from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Union
from enum import Enum
import uuid

class ElementType(str, Enum):
    TEXT = "text"
    TABLE = "table"
    IMAGE = "image"
    CODE = "code"

@dataclass
class Element:
    """Represents a specific part of a document (text, table, image, etc.)"""
    content: Any # Str for text, bytes/path for image, etc.
    type: ElementType
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Document:
    """Represents a source document."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    elements: List[Element] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_text_content(self) -> str:
        """Returns the concatenated text content of all text elements."""
        return "\n".join([e.content for e in self.elements if e.type == ElementType.TEXT])

class ChunkType(str, Enum):
    TEXT = "text"
    IMAGE = "image"

@dataclass
class Chunk:
    """Represents an indexable unit of data."""
    content: Any
    chunk_type: ChunkType
    source_document_id: str
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    embedding: Optional[List[float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def has_embedding(self) -> bool:
        return self.embedding is not None

@dataclass
class RetrievalResult:
    """Represents a search result."""
    chunk: Chunk
    score: float
    source: str = "vector" # vector, graph, hybrid
    metadata: Dict[str, Any] = field(default_factory=dict)

class QuantizationType(str, Enum):
    NONE = "none"
    BINARY = "binary"
    PRODUCT = "product"

@dataclass
class QuantizationConfig:
    """Configuration for optimization."""
    type: QuantizationType = QuantizationType.NONE
    always_ram: bool = False
    rescore: bool = True
