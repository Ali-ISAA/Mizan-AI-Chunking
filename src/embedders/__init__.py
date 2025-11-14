"""Embedding providers with lazy loading"""

from .base import BaseEmbedder

__all__ = ['BaseEmbedder', 'get_embedder']


def get_embedder(provider: str, model_name: str, dimension: int, **kwargs) -> BaseEmbedder:
    """
    Factory function to get embedder instance

    Uses lazy imports to avoid loading unnecessary dependencies.
    Only the requested embedder provider is imported and initialized.

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
        from .gemini import GeminiEmbedder
        return GeminiEmbedder(model_name=model_name, dimension=dimension, **kwargs)
    elif provider == 'openai':
        from .openai import OpenAIEmbedder
        return OpenAIEmbedder(model_name=model_name, dimension=dimension, **kwargs)
    elif provider == 'ollama':
        from .ollama import OllamaEmbedder
        return OllamaEmbedder(model_name=model_name, dimension=dimension, **kwargs)
    else:
        raise ValueError(f"Unknown embedding provider: {provider}")
