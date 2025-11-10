"""
Qdrant vector store implementation
"""

from typing import List, Dict, Optional
import uuid
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct,
    Filter, FieldCondition, MatchValue
)

from .base import BaseVectorStore
from ..utils.config import get_config


class QdrantStore(BaseVectorStore):
    """Qdrant vector store implementation"""

    def __init__(self, collection_name: str, dimension: int):
        """
        Initialize Qdrant store

        Parameters:
        -----------
        collection_name : str
            Name of the collection
        dimension : int
            Dimension of embeddings
        """
        super().__init__(collection_name, dimension)

        # Load configuration
        config = get_config()

        # Initialize Qdrant client
        try:
            if config.qdrant_api_key:
                self.client = QdrantClient(
                    url=config.qdrant_url,
                    api_key=config.qdrant_api_key
                )
            else:
                self.client = QdrantClient(url=config.qdrant_url)
        except Exception as e:
            raise ConnectionError(f"Failed to initialize Qdrant client: {str(e)}")

    def create_collection(self) -> bool:
        """
        Create collection if it doesn't exist

        Returns:
        --------
        bool
            True if created, False if already exists
        """
        try:
            # Check if collection exists
            collections = self.client.get_collections().collections
            collection_names = [col.name for col in collections]

            if self.collection_name in collection_names:
                return False

            # Create collection
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.dimension,
                    distance=Distance.COSINE
                )
            )
            return True
        except Exception as e:
            raise RuntimeError(f"Failed to create collection: {str(e)}")

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
        try:
            # Generate IDs if not provided
            if ids is None:
                ids = [str(uuid.uuid4()) for _ in range(len(texts))]

            # Prepare metadata
            if metadata is None:
                metadata = [{} for _ in range(len(texts))]

            # Prepare points
            points = []
            for i in range(len(texts)):
                # Add text to metadata
                payload = metadata[i].copy()
                payload['text'] = texts[i]

                points.append(
                    PointStruct(
                        id=ids[i],
                        vector=embeddings[i],
                        payload=payload
                    )
                )

            # Insert into Qdrant
            self.client.upsert(
                collection_name=self.collection_name,
                points=points
            )
            return True
        except Exception as e:
            raise RuntimeError(f"Failed to insert documents: {str(e)}")

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
        try:
            # Build filter if provided
            query_filter = None
            if filters:
                conditions = []
                for key, value in filters.items():
                    conditions.append(
                        FieldCondition(
                            key=key,
                            match=MatchValue(value=value)
                        )
                    )
                if conditions:
                    query_filter = Filter(must=conditions)

            # Perform search
            results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=top_k,
                query_filter=query_filter
            )

            # Format results
            formatted_results = []
            for result in results:
                payload = result.payload.copy()
                text = payload.pop('text', '')

                formatted_results.append({
                    'text': text,
                    'metadata': payload,
                    'score': result.score,
                    'id': result.id
                })

            return formatted_results
        except Exception as e:
            raise RuntimeError(f"Failed to search: {str(e)}")

    def delete_collection(self) -> bool:
        """
        Delete collection

        Returns:
        --------
        bool
            True if deleted
        """
        try:
            self.client.delete_collection(collection_name=self.collection_name)
            return True
        except Exception as e:
            raise RuntimeError(f"Failed to delete collection: {str(e)}")

    def get_count(self) -> int:
        """
        Get number of vectors in collection

        Returns:
        --------
        int
            Number of vectors
        """
        try:
            collection_info = self.client.get_collection(collection_name=self.collection_name)
            return collection_info.points_count if collection_info.points_count else 0
        except Exception as e:
            # Collection might not exist
            return 0
