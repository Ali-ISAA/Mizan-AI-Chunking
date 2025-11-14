"""Chunker implementations with lazy loading"""

from .base import BaseChunker

__all__ = ['BaseChunker', 'get_chunker']


def get_chunker(chunker_type: str, **kwargs) -> BaseChunker:
    """
    Factory function to get chunker instance

    Uses lazy imports to avoid loading unnecessary dependencies.
    Only the requested chunker is imported and initialized.

    Parameters:
    -----------
    chunker_type : str
        Type of chunker (fixed, recursive, cluster, kamradt, llm, context-aware, section)
    **kwargs
        Additional arguments for the chunker

    Returns:
    --------
    BaseChunker
        Chunker instance
    """
    chunker_type = chunker_type.lower()

    if chunker_type == 'fixed':
        from .fixed_token import FixedTokenChunker
        return FixedTokenChunker(**kwargs)
    elif chunker_type == 'recursive':
        from .recursive import RecursiveChunker
        return RecursiveChunker(**kwargs)
    elif chunker_type == 'cluster':
        from .cluster_semantic import ClusterSemanticChunker
        return ClusterSemanticChunker(**kwargs)
    elif chunker_type == 'kamradt':
        from .kamradt_semantic import KamradtSemanticChunker
        return KamradtSemanticChunker(**kwargs)
    elif chunker_type == 'llm':
        from .llm_semantic import LLMSemanticChunker
        return LLMSemanticChunker(**kwargs)
    elif chunker_type == 'context-aware' or chunker_type == 'context_aware':
        from .context_aware import ContextAwareChunker
        return ContextAwareChunker(**kwargs)
    elif chunker_type == 'section':
        from .section_based import SectionBasedChunker
        return SectionBasedChunker(**kwargs)
    else:
        raise ValueError(f"Unknown chunker type: {chunker_type}")
