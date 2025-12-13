import os
from typing import Iterator, Optional, List
import google.generativeai as genai
from polyrag.core.ports.llm_port import LLMPort

class GeminiAdapter(LLMPort):
    """Adapter for Google Gemini models via google-generativeai library."""

    def __init__(self, model_name: str = "gemini-2.5-pro", api_key: Optional[str] = None):
        """
        Initialize the Gemini adapter.

        Args:
            model_name: The name of the Gemini model to use. Defaults to "gemini-2.5-pro".
                        User can override to "gemini-1.5-pro", "gemini-3.0-pro-preview", etc.
            api_key: Google Gemini API key. If not provided, looks for GOOGLE_API_KEY env var.
        """
        self._model_name = model_name
        
        if api_key:
            genai.configure(api_key=api_key)
        elif "GOOGLE_API_KEY" in os.environ:
            genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
        else:
            # We don't raise error here, as it might be set later or user might just be initializing
            pass

        self._model = genai.GenerativeModel(model_name)

    @staticmethod
    def list_models(api_key: Optional[str] = None) -> List[str]:
        """
        List available Gemini models that support content generation.
        """
        if api_key:
            genai.configure(api_key=api_key)
        elif "GOOGLE_API_KEY" in os.environ:
            genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
            
        try:
            models = []
            for m in genai.list_models():
                if 'generateContent' in m.supported_generation_methods:
                    models.append(m.name.replace('models/', ''))
            return sorted(models)
        except Exception as e:
            # Fallback or empty list if config is missing/invalid
            print(f"Error listing models: {e}")
            return []

    def generate(self, prompt: str, system_prompt: Optional[str] = None, **kwargs) -> str:
        """
        Generates a text response for the given prompt.

        Args:
            prompt: The user prompt.
            system_prompt: Optional system prompt. 
                           Note: Gemini supports system instructions at model init or via specific API usage.
                           To keep it simple per request, we might prepend it or use the proper config if we re-init.
            **kwargs: Additional parameters (e.g., generation_config).

        Returns:
            The generated text response.
        """
        # Gemini handles system prompts best via system_instruction in model init, 
        # but since we reuse the model, we can try prepending or checking if we can update it.
        # For simplicity and compatibility, prepending is often safest if strict system instruction object isn't needed.
        # However, new SDK allows `system_instruction` in GenerativeModel. 
        # Making a new instance for each call is cheap reference-wise if we want to enforce system prompt.
        
        model_to_use = self._model
        if system_prompt:
             model_to_use = genai.GenerativeModel(self._model_name, system_instruction=system_prompt)

        response = model_to_use.generate_content(prompt, **kwargs)
        return response.text

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
        model_to_use = self._model
        if system_prompt:
             model_to_use = genai.GenerativeModel(self._model_name, system_instruction=system_prompt)

        response = model_to_use.generate_content(prompt, stream=True, **kwargs)
        for chunk in response:
            if chunk.text:
                yield chunk.text

    @property
    def model_name(self) -> str:
        """Returns the name of the model being used."""
        return self._model_name

    @property
    def supports_vision(self) -> bool:
        """Returns True if the model supports image inputs."""
        # Most Gemini Pro models support vision
        return "pro" in self._model_name.lower() or "flash" in self._model_name.lower()
