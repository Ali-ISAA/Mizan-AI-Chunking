"""LLM providers"""

from .base import BaseLLM
from .gemini import GeminiLLM
from .openai import OpenAILLM
from .ollama import OllamaLLM
from .litellm import LiteLLM

__all__ = ['BaseLLM', 'GeminiLLM', 'OpenAILLM', 'OllamaLLM', 'LiteLLM']


def get_llm(provider: str, model_name: str, **kwargs) -> BaseLLM:
    """
    Factory function to get LLM instance

    Parameters:
    -----------
    provider : str
        Provider name (gemini, openai, ollama, litellm)
    model_name : str
        Model name
    **kwargs
        Additional arguments for the LLM

    Returns:
    --------
    BaseLLM
        LLM instance
    """
    provider = provider.lower()

    if provider == 'gemini':
        return GeminiLLM(model_name=model_name, **kwargs)
    elif provider == 'openai':
        return OpenAILLM(model_name=model_name, **kwargs)
    elif provider == 'ollama':
        return OllamaLLM(model_name=model_name, **kwargs)
    elif provider == 'litellm':
        return LiteLLM(model_name=model_name, **kwargs)
    else:
        raise ValueError(f"Unknown LLM provider: {provider}")
