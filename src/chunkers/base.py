"""
Base class for chunkers
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Optional
import tiktoken


class BaseChunker(ABC):
    """Abstract base class for all chunkers"""

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50):
        """
        Initialize chunker

        Parameters:
        -----------
        chunk_size : int
            Target chunk size in tokens
        chunk_overlap : int
            Overlap between chunks in tokens
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.encoding = tiktoken.get_encoding("cl100k_base")

    @abstractmethod
    def chunk(self, text: str, metadata: Optional[Dict] = None) -> List[Dict]:
        """
        Chunk text into smaller pieces

        Parameters:
        -----------
        text : str
            Text to chunk
        metadata : Dict, optional
            Additional metadata to attach to chunks

        Returns:
        --------
        List[Dict]
            List of chunks with metadata
            Each dict has: {'text': str, 'metadata': Dict, 'tokens': int}
        """
        pass

    def count_tokens(self, text: str) -> int:
        """
        Count tokens in text

        Parameters:
        -----------
        text : str
            Text to count

        Returns:
        --------
        int
            Number of tokens
        """
        return len(self.encoding.encode(text))

    def _create_chunk_dict(self, text: str, chunk_index: int,
                          metadata: Optional[Dict] = None) -> Dict:
        """
        Create standardized chunk dictionary

        Parameters:
        -----------
        text : str
            Chunk text
        chunk_index : int
            Index of chunk
        metadata : Dict, optional
            Additional metadata

        Returns:
        --------
        Dict
            Chunk dictionary
        """
        chunk_metadata = metadata.copy() if metadata else {}
        chunk_metadata['chunk_index'] = chunk_index
        chunk_metadata['tokens'] = self.count_tokens(text)

        return {
            'text': text,
            'metadata': chunk_metadata,
            'tokens': chunk_metadata['tokens']
        }

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(size={self.chunk_size}, overlap={self.chunk_overlap})"
