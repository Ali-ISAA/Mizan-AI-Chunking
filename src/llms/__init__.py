"""LLM providers with lazy loading"""

from .base import BaseLLM

__all__ = ['BaseLLM', 'get_llm']


def get_llm(provider: str, model_name: str, **kwargs) -> BaseLLM:
    """
    Factory function to get LLM instance

    Uses lazy imports to avoid loading unnecessary dependencies.
    Only the requested LLM provider is imported and initialized.

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
        from .gemini import GeminiLLM
        return GeminiLLM(model_name=model_name, **kwargs)
    elif provider == 'openai':
        from .openai import OpenAILLM
        return OpenAILLM(model_name=model_name, **kwargs)
    elif provider == 'ollama':
        from .ollama import OllamaLLM
        return OllamaLLM(model_name=model_name, **kwargs)
    elif provider == 'litellm':
        from .litellm import LiteLLM
        return LiteLLM(model_name=model_name, **kwargs)
    else:
        raise ValueError(f"Unknown LLM provider: {provider}")
