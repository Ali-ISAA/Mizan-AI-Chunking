"""
Kamradt semantic chunker - uses similarity between consecutive sentences to find breakpoints
"""

import re
from typing import List, Dict, Optional
import numpy as np

from .base import BaseChunker
from ..embedders import get_embedder
from ..utils.config import get_config


class KamradtSemanticChunker(BaseChunker):
    """Uses sentence similarity analysis to find natural breakpoints"""

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 0,
                 breakpoint_percentile: int = 95,
                 embedder_provider: Optional[str] = None,
                 embedder_model: Optional[str] = None):
        """
        Initialize Kamradt semantic chunker

        Parameters:
        -----------
        chunk_size : int
            Target chunk size (used for reference)
        chunk_overlap : int
            Overlap between chunks
        breakpoint_percentile : int
            Percentile threshold for determining breakpoints (default: 95)
        embedder_provider : str, optional
            Embedder provider (loads from config if None)
        embedder_model : str, optional
            Embedder model name (loads from config if None)
        """
        super().__init__(chunk_size, chunk_overlap)

        self.breakpoint_percentile = breakpoint_percentile

        # Get config
        config = get_config()
        self.embedder_provider = embedder_provider or config.embedding_provider
        self.embedder_model = embedder_model or config.embedding_model
        self.embedding_dimension = config.embedding_dimension

        # Initialize embedder
        self.embedder = get_embedder(
            self.embedder_provider,
            self.embedder_model,
            self.embedding_dimension
        )

    def chunk(self, text: str, metadata: Optional[Dict] = None) -> List[Dict]:
        """
        Chunk text using similarity-based breakpoint detection

        Parameters:
        -----------
        text : str
            Text to chunk
        metadata : Dict, optional
            Additional metadata

        Returns:
        --------
        List[Dict]
            List of chunks
        """
        print(f"Using Kamradt semantic chunking with {self.embedder_provider}")

        # Split into sentences
        sentences = self._split_sentences(text)

        if len(sentences) < 2:
            return [self._create_chunk_dict(text, 0, metadata)]

        print(f"  Analyzing {len(sentences)} sentences")

        # Generate embeddings
        print(f"  Generating embeddings...")
        embeddings = self.embedder.embed_batch(sentences)

        # Calculate similarities between consecutive sentences
        similarities = []
        for i in range(len(embeddings) - 1):
            similarity = self._cosine_similarity(embeddings[i], embeddings[i + 1])
            similarities.append(similarity)

        # Find breakpoints (low similarity = topic change)
        breakpoints = self._find_breakpoints(similarities)

        print(f"  Found {len(breakpoints)} semantic breakpoints")

        # Create chunks from breakpoints
        chunks = []
        start_idx = 0

        for breakpoint in breakpoints:
            chunk_sentences = sentences[start_idx:breakpoint + 1]
            chunk_text = ' '.join(chunk_sentences)

            chunks.append(self._create_chunk_dict(chunk_text, len(chunks), metadata))
            start_idx = breakpoint + 1

        # Add final chunk
        if start_idx < len(sentences):
            chunk_sentences = sentences[start_idx:]
            chunk_text = ' '.join(chunk_sentences)
            chunks.append(self._create_chunk_dict(chunk_text, len(chunks), metadata))

        print(f"Created {len(chunks)} semantic chunks")
        return chunks

    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)

        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)

    def _find_breakpoints(self, similarities: List[float]) -> List[int]:
        """Find breakpoints where similarity drops significantly"""
        if not similarities:
            return []

        # Calculate threshold using percentile
        threshold = np.percentile(similarities, 100 - self.breakpoint_percentile)

        # Find indices where similarity is below threshold
        breakpoints = []
        for i, sim in enumerate(similarities):
            if sim < threshold:
                breakpoints.append(i)

        return breakpoints
