"""
Base class for vector stores
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Tuple


class BaseVectorStore(ABC):
    """Abstract base class for vector stores"""

    def __init__(self, collection_name: str, dimension: int):
        """
        Initialize vector store

        Parameters:
        -----------
        collection_name : str
            Name of the collection/table/index
        dimension : int
            Dimension of embeddings
        """
        self.collection_name = collection_name
        self.dimension = dimension

    @abstractmethod
    def create_collection(self) -> bool:
        """
        Create collection/table/index if it doesn't exist

        Returns:
        --------
        bool
            True if created, False if already exists
        """
        pass

    @abstractmethod
    def insert(self, texts: List[str], embeddings: List[List[float]],
               metadata: Optional[List[Dict]] = None, ids: Optional[List[str]] = None) -> bool:
        """
        Insert texts with embeddings

        Parameters:
        -----------
        texts : List[str]
            List of text chunks
        embeddings : List[List[float]]
            List of embeddings
        metadata : List[Dict], optional
            List of metadata dictionaries
        ids : List[str], optional
            List of IDs (auto-generated if None)

        Returns:
        --------
        bool
            True if successful
        """
        pass

    @abstractmethod
    def search(self, query_embedding: List[float], top_k: int = 10,
               filters: Optional[Dict] = None) -> List[Dict]:
        """
        Search for similar vectors

        Parameters:
        -----------
        query_embedding : List[float]
            Query embedding
        top_k : int
            Number of results to return
        filters : Dict, optional
            Metadata filters

        Returns:
        --------
        List[Dict]
            List of results with text, metadata, and score
        """
        pass

    @abstractmethod
    def delete_collection(self) -> bool:
        """
        Delete collection/table/index

        Returns:
        --------
        bool
            True if deleted
        """
        pass

    @abstractmethod
    def get_count(self) -> int:
        """
        Get number of vectors in collection

        Returns:
        --------
        int
            Number of vectors
        """
        pass

    def scroll(self, offset: int = 0, limit: int = 50,
               filters: Optional[Dict] = None) -> Tuple[List[Dict], int]:
        """
        Paginate through all vectors in collection.

        Parameters:
        -----------
        offset : int
            Number of records to skip
        limit : int
            Maximum number of records to return
        filters : Dict, optional
            Metadata filters

        Returns:
        --------
        Tuple[List[Dict], int]
            (results, total_count) - results contain id, text, metadata
        """
        raise NotImplementedError(
            f"scroll() not implemented for {self.__class__.__name__}. "
            "This store does not support chunk browsing yet."
        )

    def get_by_id(self, point_id: str) -> Optional[Dict]:
        """
        Get a single vector by its ID.

        Parameters:
        -----------
        point_id : str
            The vector/point ID

        Returns:
        --------
        Optional[Dict]
            Vector data with id, text, metadata, or None if not found
        """
        raise NotImplementedError(
            f"get_by_id() not implemented for {self.__class__.__name__}. "
            "This store does not support individual chunk retrieval yet."
        )

    def get_metadata_fields(self, sample_size: int = 100) -> List[Dict]:
        """
        Get available metadata fields by sampling vectors.

        Parameters:
        -----------
        sample_size : int
            Number of vectors to sample for field discovery

        Returns:
        --------
        List[Dict]
            List of metadata field info with name, type, sample_values
        """
        try:
            results, _ = self.scroll(limit=sample_size)
        except NotImplementedError:
            return []

        fields = {}
        for result in results:
            for key, val in result.get('metadata', {}).items():
                if key not in fields:
                    fields[key] = {
                        'name': key,
                        'type': type(val).__name__,
                        'samples': set()
                    }
                if len(fields[key]['samples']) < 5:
                    fields[key]['samples'].add(str(val)[:50])

        return [
            {
                'name': f['name'],
                'type': f['type'],
                'sample_values': list(f['samples'])
            }
            for f in fields.values()
        ]

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(collection={self.collection_name}, dim={self.dimension})"
