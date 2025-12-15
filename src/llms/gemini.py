"""
Google Gemini LLM provider
"""

import time
from typing import Optional, List
import google.generativeai as genai

from .base import BaseLLM
from ..utils.api_key_manager import APIKeyManager
from ..utils.config import get_config


class GeminiLLM(BaseLLM):
    """Google Gemini LLM implementation"""

    def __init__(self, model_name: str = "gemini-2.0-flash-lite",
                 temperature: float = 0.2, max_tokens: int = 2048,
                 api_keys: Optional[List[str]] = None):
        """
        Initialize Gemini LLM

        Parameters:
        -----------
        model_name : str
            Gemini model name
        temperature : float
            Temperature for generation
        max_tokens : int
            Maximum tokens in response
        api_keys : List[str], optional
            List of API keys for rotation (loads from config if None)
        """
        super().__init__(model_name, temperature, max_tokens)

        # Get API keys
        if api_keys is None:
            config = get_config()
            api_keys = config.get_gemini_keys()

        if not api_keys:
            raise ValueError("No Gemini API keys provided")

        # Store config for model recreation
        self._model_name = model_name
        self._temperature = temperature
        self._max_tokens = max_tokens

        # Initialize key manager with callback to reconfigure
        self.key_manager = APIKeyManager(api_keys, on_key_change=self._reconfigure_client)

        # Configure with first key and create model
        self._reconfigure_client(self.key_manager.get_current_key())

    def _reconfigure_client(self, api_key: str):
        """Reconfigure genai with new API key and recreate model"""
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(
            self._model_name,
            generation_config=genai.types.GenerationConfig(
                temperature=self._temperature,
                max_output_tokens=self._max_tokens
            )
        )

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """
        Generate text from prompt

        Parameters:
        -----------
        prompt : str
            User prompt
        system_prompt : str, optional
            System prompt (prepended to prompt)

        Returns:
        --------
        str
            Generated text
        """
        full_prompt = prompt
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n{prompt}"

        response = self.model.generate_content(full_prompt)
        return response.text

    def generate_with_retry(self, prompt: str, system_prompt: Optional[str] = None,
                           max_retries: int = None) -> str:
        """
        Generate text with automatic retry on rate limits and invalid keys

        Parameters:
        -----------
        prompt : str
            User prompt
        system_prompt : str, optional
            System prompt
        max_retries : int, optional
            Maximum number of retries (default: number of keys * 2)

        Returns:
        --------
        str
            Generated text
        """
        def generate_func():
            return self.generate(prompt, system_prompt)

        return self.key_manager.execute_with_retry(
            generate_func,
            max_retries=max_retries
        )
