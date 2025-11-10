"""Chunker implementations"""

from .base import BaseChunker
from .fixed_token import FixedTokenChunker
from .recursive import RecursiveChunker
from .cluster_semantic import ClusterSemanticChunker
from .kamradt_semantic import KamradtSemanticChunker
from .llm_semantic import LLMSemanticChunker
from .context_aware import ContextAwareChunker
from .section_based import SectionBasedChunker

__all__ = [
    'BaseChunker',
    'FixedTokenChunker',
    'RecursiveChunker',
    'ClusterSemanticChunker',
    'KamradtSemanticChunker',
    'LLMSemanticChunker',
    'ContextAwareChunker',
    'SectionBasedChunker'
]


def get_chunker(chunker_type: str, **kwargs) -> BaseChunker:
    """
    Factory function to get chunker instance

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
        return FixedTokenChunker(**kwargs)
    elif chunker_type == 'recursive':
        return RecursiveChunker(**kwargs)
    elif chunker_type == 'cluster':
        return ClusterSemanticChunker(**kwargs)
    elif chunker_type == 'kamradt':
        return KamradtSemanticChunker(**kwargs)
    elif chunker_type == 'llm':
        return LLMSemanticChunker(**kwargs)
    elif chunker_type == 'context-aware' or chunker_type == 'context_aware':
        return ContextAwareChunker(**kwargs)
    elif chunker_type == 'section':
        return SectionBasedChunker(**kwargs)
    else:
        raise ValueError(f"Unknown chunker type: {chunker_type}")
