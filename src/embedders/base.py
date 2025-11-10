"""
Base class for embedding providers
"""

from abc import ABC, abstractmethod
from typing import List, Union


class BaseEmbedder(ABC):
    """Abstract base class for embedding providers"""

    def __init__(self, model_name: str, dimension: int):
        """
        Initialize embedder

        Parameters:
        -----------
        model_name : str
            Name of the embedding model
        dimension : int
            Embedding dimension
        """
        self.model_name = model_name
        self.dimension = dimension

    @abstractmethod
    def embed(self, text: Union[str, List[str]]) -> Union[List[float], List[List[float]]]:
        """
        Generate embeddings for text(s)

        Parameters:
        -----------
        text : str or List[str]
            Text or list of texts to embed

        Returns:
        --------
        List[float] or List[List[float]]
            Single embedding or list of embeddings
        """
        pass

    @abstractmethod
    def embed_batch(self, texts: List[str], batch_size: int = 100) -> List[List[float]]:
        """
        Generate embeddings for batch of texts

        Parameters:
        -----------
        texts : List[str]
            List of texts to embed
        batch_size : int
            Batch size for processing

        Returns:
        --------
        List[List[float]]
            List of embeddings
        """
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model={self.model_name}, dim={self.dimension})"
