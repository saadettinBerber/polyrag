import requests
import json
from typing import Iterator, Optional

from polyrag.core.ports.llm_port import LLMPort


class OllamaAdapter(LLMPort):
    """Adapter for Ollama local LLM API."""

    # Models known to support vision
    VISION_MODELS = ["llava", "llava-llama3", "bakllava", "moondream"]

    def __init__(self, model: str = "llama3.2", base_url: str = "http://localhost:11434"):
        """
        Initialize the Ollama adapter.

        Args:
            model: The name of the Ollama model to use.
            base_url: The base URL for the Ollama API.
        """
        self._model = model
        self._base_url = base_url.rstrip("/")

    def generate(self, prompt: str, system_prompt: Optional[str] = None, **kwargs) -> str:
        """
        Generates a text response for the given prompt.

        Args:
            prompt: The user prompt.
            system_prompt: Optional system prompt.
            **kwargs: Additional parameters (e.g., images for vision models).

        Returns:
            The generated text response.
        """
        payload = {
            "model": self._model,
            "prompt": prompt,
            "stream": False,
        }

        if system_prompt:
            payload["system"] = system_prompt

        # Support for images (vision models)
        if "images" in kwargs:
            payload["images"] = kwargs["images"]

        response = requests.post(
            f"{self._base_url}/api/generate",
            json=payload,
            timeout=120
        )
        response.raise_for_status()
        return response.json().get("response", "")

    def generate_stream(self, prompt: str, system_prompt: Optional[str] = None, **kwargs) -> Iterator[str]:
        """
        Generates a streaming text response.

        Args:
            prompt: The user prompt.
            system_prompt: Optional system prompt.
            **kwargs: Additional parameters.

        Yields:
            Chunks of the generated text response.
        """
        payload = {
            "model": self._model,
            "prompt": prompt,
            "stream": True,
        }

        if system_prompt:
            payload["system"] = system_prompt

        if "images" in kwargs:
            payload["images"] = kwargs["images"]

        with requests.post(
            f"{self._base_url}/api/generate",
            json=payload,
            stream=True,
            timeout=120
        ) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if line:
                    data = json.loads(line)
                    if "response" in data:
                        yield data["response"]

    @property
    def model_name(self) -> str:
        """Returns the name of the model being used."""
        return self._model

    @property
    def supports_vision(self) -> bool:
        """Returns True if the model supports image inputs."""
        return any(vm in self._model.lower() for vm in self.VISION_MODELS)
