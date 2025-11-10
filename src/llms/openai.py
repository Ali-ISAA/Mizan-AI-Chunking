"""
OpenAI LLM provider (also compatible with vLLM, OpenRouter, etc.)
"""

import time
from typing import Optional

from .base import BaseLLM
from ..utils.config import get_config


class OpenAILLM(BaseLLM):
    """OpenAI LLM implementation (compatible with OpenAI API format)"""

    def __init__(self, model_name: str = "gpt-4o-mini",
                 temperature: float = 0.2, max_tokens: int = 2048,
                 api_key: Optional[str] = None, base_url: Optional[str] = None):
        """
        Initialize OpenAI LLM

        Parameters:
        -----------
        model_name : str
            Model name
        temperature : float
            Temperature for generation
        max_tokens : int
            Maximum tokens in response
        api_key : str, optional
            API key (loads from config if None)
        base_url : str, optional
            Base URL for API (for vLLM, etc.)
        """
        super().__init__(model_name, temperature, max_tokens)

        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("openai package required. Install: pip install openai")

        # Get config
        config = get_config()
        self.api_key = api_key or config.openai_api_key
        self.base_url = base_url or config.openai_base_url

        if not self.api_key:
            raise ValueError("OpenAI API key not provided")

        # Initialize client
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )

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

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )

        return response.choices[0].message.content

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
                    time.sleep(2 ** attempt)  # Exponential backoff
                    continue
                else:
                    break

        raise last_exception
