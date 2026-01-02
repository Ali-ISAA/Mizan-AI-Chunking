"""
Ollama LLM provider for local models
"""

import time
from typing import Optional

from .base import BaseLLM
from ..utils.config import get_config


class OllamaLLM(BaseLLM):
    """Ollama LLM implementation"""

    def __init__(self, model_name: str = "llama3.2",
                 temperature: float = 0.2, max_tokens: int = 2048,
                 base_url: Optional[str] = None, **kwargs):
        """
        Initialize Ollama LLM

        Parameters:
        -----------
        model_name : str
            Ollama model name
        temperature : float
            Temperature for generation
        max_tokens : int
            Maximum tokens in response
        base_url : str, optional
            Ollama server URL (default: http://localhost:11434)
        **kwargs
            Additional arguments (ignored - Ollama doesn't use api_key etc.)
        """
        # Convert temperature to float to handle SQLAlchemy Decimal values
        super().__init__(model_name, float(temperature), max_tokens)

        try:
            import ollama
            self._ollama = ollama
        except ImportError:
            raise ImportError("ollama package required. Install: pip install ollama")

        # Use provided base_url or fall back to config/default
        if base_url:
            self.base_url = base_url
        else:
            try:
                config = get_config()
                self.base_url = config.ollama_base_url
            except (ValueError, Exception):
                # Fallback to default if config validation fails
                self.base_url = "http://localhost:11434"

        # Initialize sync client for non-streaming operations
        self.client = ollama.Client(host=self.base_url)
        # Initialize async client for streaming operations
        self.async_client = ollama.AsyncClient(host=self.base_url)

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """
        Generate text from prompt

        Parameters:
        -----------
        prompt : str
            User prompt
        system_prompt : str, optional
            System prompt

        Returns:
        --------
        str
            Generated text
        """
        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.append({"role": "user", "content": prompt})

        response = self.client.chat(
            model=self.model_name,
            messages=messages,
            options={
                "temperature": self.temperature,
                "num_predict": self.max_tokens
            }
        )

        return response['message']['content']

    async def generate_stream(self, prompt: str, system_prompt: Optional[str] = None):
        """
        Generate text from prompt with streaming.

        Parameters:
        -----------
        prompt : str
            User prompt
        system_prompt : str, optional
            System prompt

        Yields:
        -------
        str
            Generated text tokens
        """
        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.append({"role": "user", "content": prompt})

        # Use async client for true non-blocking streaming
        stream = await self.async_client.chat(
            model=self.model_name,
            messages=messages,
            options={
                "temperature": self.temperature,
                "num_predict": self.max_tokens
            },
            stream=True
        )

        async for chunk in stream:
            if 'message' in chunk and 'content' in chunk['message']:
                yield chunk['message']['content']

    def generate_with_retry(self, prompt: str, system_prompt: Optional[str] = None,
                           max_retries: int = 3) -> str:
        """
        Generate text with automatic retry on errors

        Parameters:
        -----------
        prompt : str
            User prompt
        system_prompt : str, optional
            System prompt
        max_retries : int
            Maximum number of retries

        Returns:
        --------
        str
            Generated text
        """
        last_exception = None

        for attempt in range(max_retries):
            try:
                return self.generate(prompt, system_prompt)
            except Exception as e:
                last_exception = e
                if attempt < max_retries - 1:
                    print(f"  Retry attempt {attempt + 1}/{max_retries}")
                    time.sleep(1)
                    continue
                else:
                    break

        raise last_exception
