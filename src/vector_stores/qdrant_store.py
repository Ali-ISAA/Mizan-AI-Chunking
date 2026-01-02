"""
Qdrant vector store implementation
"""

from typing import List, Dict, Optional, Tuple
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

    def __init__(self, collection_name: str, dimension: int, url: str = None, api_key: str = None, **kwargs):
        """
        Initialize Qdrant store

        Parameters:
        -----------
        collection_name : str
            Name of the collection
        dimension : int
            Dimension of embeddings
        url : str, optional
            Qdrant server URL (overrides config if provided)
        api_key : str, optional
            Qdrant API key (overrides config if provided)
        **kwargs
            Additional arguments (ignored for compatibility)
        """
        super().__init__(collection_name, dimension)

        # Use provided values or fall back to config/defaults
        if url:
            qdrant_url = url
            qdrant_api_key = api_key
        else:
            try:
                config = get_config()
                qdrant_url = config.qdrant_url
                qdrant_api_key = api_key or config.qdrant_api_key
            except (ValueError, Exception):
                # Fallback to defaults if config validation fails
                qdrant_url = "http://localhost:6333"
                qdrant_api_key = api_key

        # Initialize Qdrant client
        try:
            if qdrant_api_key:
                self.client = QdrantClient(
                    url=qdrant_url,
                    api_key=qdrant_api_key
                )
            else:
                self.client = QdrantClient(url=qdrant_url)
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
                query_filter=query_filter,
                with_payload=True,
            )

            # Format results
            formatted_results = []
            for point in results:
                payload = point.payload.copy() if point.payload else {}
                text = payload.pop('text', '')

                formatted_results.append({
                    'text': text,
                    'metadata': payload,
                    'score': point.score,
                    'id': str(point.id)
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
            (results, total_count)
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

            # Get total count
            total = self.get_count()

            # Qdrant scroll returns all points, we need to handle offset/limit manually
            # For better performance with large collections, use scroll with offset
            records, _ = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=query_filter,
                limit=limit,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )

            # Format results
            formatted_results = []
            for record in records:
                payload = record.payload.copy() if record.payload else {}
                text = payload.pop('text', '')

                formatted_results.append({
                    'id': str(record.id),
                    'text': text,
                    'metadata': payload,
                })

            return formatted_results, total
        except Exception as e:
            raise RuntimeError(f"Failed to scroll: {str(e)}")

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
        try:
            results = self.client.retrieve(
                collection_name=self.collection_name,
                ids=[point_id],
                with_payload=True,
                with_vectors=False,
            )

            if not results:
                return None

            record = results[0]
            payload = record.payload.copy() if record.payload else {}
            text = payload.pop('text', '')

            return {
                'id': str(record.id),
                'text': text,
                'metadata': payload,
            }
        except Exception as e:
            raise RuntimeError(f"Failed to get point: {str(e)}")
