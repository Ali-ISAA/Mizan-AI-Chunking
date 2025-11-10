"""
Cluster semantic chunker - uses clustering on sentence embeddings
"""

import re
from typing import List, Dict, Optional
import numpy as np

from .base import BaseChunker
from ..embedders import get_embedder
from ..utils.config import get_config


class ClusterSemanticChunker(BaseChunker):
    """Uses K-means clustering on sentence embeddings to create semantic chunks"""

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 0,
                 num_clusters: Optional[int] = None,
                 embedder_provider: Optional[str] = None,
                 embedder_model: Optional[str] = None):
        """
        Initialize cluster semantic chunker

        Parameters:
        -----------
        chunk_size : int
            Target chunk size (used for cluster count estimation)
        chunk_overlap : int
            Overlap between chunks
        num_clusters : int, optional
            Number of clusters (auto-calculated if None)
        embedder_provider : str, optional
            Embedder provider (loads from config if None)
        embedder_model : str, optional
            Embedder model name (loads from config if None)
        """
        super().__init__(chunk_size, chunk_overlap)

        self.num_clusters = num_clusters

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
        Chunk text using clustering

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
        try:
            from sklearn.cluster import KMeans
        except ImportError:
            raise ImportError("scikit-learn required. Install: pip install scikit-learn")

        print(f"Using cluster semantic chunking with {self.embedder_provider}")

        # Split into sentences
        sentences = self._split_sentences(text)

        if len(sentences) < 2:
            return [self._create_chunk_dict(text, 0, metadata)]

        # Auto-calculate clusters if not specified
        num_clusters = self.num_clusters
        if num_clusters is None:
            total_tokens = self.count_tokens(text)
            num_clusters = max(2, min(len(sentences), total_tokens // self.chunk_size))

        print(f"  Splitting {len(sentences)} sentences into {num_clusters} clusters")

        # Generate embeddings
        print(f"  Generating embeddings...")
        embeddings = self.embedder.embed_batch(sentences)

        # Cluster
        kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(embeddings)

        # Group sentences by cluster
        clusters = {}
        for i, (sentence, label) in enumerate(zip(sentences, labels)):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append((i, sentence))

        # Create chunks from clusters (sorted by first sentence index)
        chunks = []
        for cluster_id in sorted(clusters.keys(), key=lambda x: clusters[x][0][0]):
            cluster_sentences = [sent for _, sent in sorted(clusters[cluster_id])]
            chunk_text = ' '.join(cluster_sentences)

            chunks.append(self._create_chunk_dict(chunk_text, len(chunks), metadata))

        print(f"Created {len(chunks)} semantic clusters")
        return chunks

    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        # Simple sentence splitter
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]
