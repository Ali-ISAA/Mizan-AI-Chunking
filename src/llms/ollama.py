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
                 base_url: Optional[str] = None):
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
        """
        super().__init__(model_name, temperature, max_tokens)

        try:
            import ollama
        except ImportError:
            raise ImportError("ollama package required. Install: pip install ollama")

        # Get config
        config = get_config()
        self.base_url = base_url or config.ollama_base_url

        # Initialize client
        self.client = ollama.Client(host=self.base_url)

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
