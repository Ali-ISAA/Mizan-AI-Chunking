"""Embedding providers"""

from .base import BaseEmbedder
from .gemini import GeminiEmbedder
from .openai import OpenAIEmbedder
from .ollama import OllamaEmbedder

__all__ = ['BaseEmbedder', 'GeminiEmbedder', 'OpenAIEmbedder', 'OllamaEmbedder']


def get_embedder(provider: str, model_name: str, dimension: int, **kwargs) -> BaseEmbedder:
    """
    Factory function to get embedder instance

    Parameters:
    -----------
    provider : str
        Provider name (gemini, openai, ollama)
    model_name : str
        Model name
    dimension : int
        Embedding dimension
    **kwargs
        Additional arguments for the embedder

    Returns:
    --------
    BaseEmbedder
        Embedder instance
    """
    provider = provider.lower()

    if provider == 'gemini':
        return GeminiEmbedder(model_name=model_name, dimension=dimension, **kwargs)
    elif provider == 'openai':
        return OpenAIEmbedder(model_name=model_name, dimension=dimension, **kwargs)
    elif provider == 'ollama':
        return OllamaEmbedder(model_name=model_name, dimension=dimension, **kwargs)
    else:
        raise ValueError(f"Unknown embedding provider: {provider}")
