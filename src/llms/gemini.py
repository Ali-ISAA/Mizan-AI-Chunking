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

        # Initialize key manager
        self.key_manager = APIKeyManager(api_keys)

        # Configure with first key
        genai.configure(api_key=self.key_manager.get_current_key())

        # Initialize model
        self.model = genai.GenerativeModel(
            model_name,
            generation_config=genai.types.GenerationConfig(
                temperature=temperature,
                max_output_tokens=max_tokens
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
                           max_retries: int = 3) -> str:
        """
        Generate text with automatic retry on rate limits

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
        def generate_func():
            # Reconfigure with current key before each attempt
            self._reconfigure_key()
            return self.generate(prompt, system_prompt)

        return self.key_manager.execute_with_retry(
            generate_func,
            max_retries=max_retries
        )

    def _reconfigure_key(self):
        """Reconfigure with current key"""
        genai.configure(api_key=self.key_manager.get_current_key())
