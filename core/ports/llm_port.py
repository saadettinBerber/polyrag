from abc import ABC, abstractmethod
from typing import Iterator, Optional, List

class LLMPort(ABC):
    """Abstract interface for Large Language Models."""

    @abstractmethod
    def generate(self, prompt: str, system_prompt: Optional[str] = None, **kwargs) -> str:
        """Generates a text response for the given prompt."""
        pass

    @abstractmethod
    def generate_stream(self, prompt: str, system_prompt: Optional[str] = None, **kwargs) -> Iterator[str]:
        """Generates a streaming text response."""
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Returns the name of the model being used."""
        pass
    
    @property
    @abstractmethod
    def supports_vision(self) -> bool:
        """Returns True if the model supports image inputs."""
        pass
