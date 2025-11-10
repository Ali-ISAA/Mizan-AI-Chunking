"""
Base class for LLM providers
"""

from abc import ABC, abstractmethod
from typing import Optional


class BaseLLM(ABC):
    """Abstract base class for LLM providers"""

    def __init__(self, model_name: str, temperature: float = 0.2, max_tokens: int = 2048):
        """
        Initialize LLM

        Parameters:
        -----------
        model_name : str
            Name of the model
        temperature : float
            Temperature for generation
        max_tokens : int
            Maximum tokens in response
        """
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens

    @abstractmethod
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
        pass

    @abstractmethod
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
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model={self.model_name})"
