"""
Pinecone vector store implementation
"""

from typing import List, Dict, Optional
import uuid
from pinecone import Pinecone, ServerlessSpec

from .base import BaseVectorStore
from ..utils.config import get_config


class PineconeStore(BaseVectorStore):
    """Pinecone vector store implementation"""

    def __init__(self, collection_name: str, dimension: int):
        """
        Initialize Pinecone store

        Parameters:
        -----------
        collection_name : str
            Name of the index
        dimension : int
            Dimension of embeddings
        """
        super().__init__(collection_name, dimension)

        # Load configuration
        config = get_config()

        if not config.pinecone_api_key:
            raise ValueError("Pinecone API key is required")

        # Initialize Pinecone client
        try:
            self.pc = Pinecone(api_key=config.pinecone_api_key)
            self.index_name = config.pinecone_index or collection_name
            self.environment = config.pinecone_environment
            self.index = None
        except Exception as e:
            raise ConnectionError(f"Failed to initialize Pinecone client: {str(e)}")

    def create_collection(self) -> bool:
        """
        Create index if it doesn't exist

        Returns:
        --------
        bool
            True if created, False if already exists
        """
        try:
            # Check if index exists
            existing_indexes = [index.name for index in self.pc.list_indexes()]

            if self.index_name in existing_indexes:
                self.index = self.pc.Index(self.index_name)
                return False

            # Create index
            # Use serverless spec (adjust cloud and region as needed)
            self.pc.create_index(
                name=self.index_name,
                dimension=self.dimension,
                metric='cosine',
                spec=ServerlessSpec(
                    cloud='aws',
                    region=self.environment or 'us-east-1'
                )
            )

            # Wait for index to be ready
            import time
            time.sleep(5)

            self.index = self.pc.Index(self.index_name)
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
            if self.index is None:
                self.create_collection()

            # Generate IDs if not provided
            if ids is None:
                ids = [str(uuid.uuid4()) for _ in range(len(texts))]

            # Prepare metadata
            if metadata is None:
                metadata = [{} for _ in range(len(texts))]

            # Prepare vectors for upsert
            vectors = []
            for i in range(len(texts)):
                # Add text to metadata
                meta = metadata[i].copy()
                meta['text'] = texts[i]

                vectors.append({
                    'id': ids[i],
                    'values': embeddings[i],
                    'metadata': meta
                })

            # Upsert in batches (Pinecone recommends batch size of 100)
            batch_size = 100
            for i in range(0, len(vectors), batch_size):
                batch = vectors[i:i + batch_size]
                self.index.upsert(vectors=batch)

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
            if self.index is None:
                raise ValueError("Index not initialized. Call create_collection() first.")

            # Build filter if provided
            pinecone_filter = None
            if filters:
                # Convert to Pinecone filter format
                pinecone_filter = {}
                for key, value in filters.items():
                    pinecone_filter[key] = {"$eq": value}

            # Perform search
            results = self.index.query(
                vector=query_embedding,
                top_k=top_k,
                filter=pinecone_filter,
                include_metadata=True
            )

            # Format results
            formatted_results = []
            for match in results.matches:
                metadata = match.metadata.copy() if match.metadata else {}
                text = metadata.pop('text', '')

                formatted_results.append({
                    'text': text,
                    'metadata': metadata,
                    'score': match.score,
                    'id': match.id
                })

            return formatted_results
        except Exception as e:
            raise RuntimeError(f"Failed to search: {str(e)}")

    def delete_collection(self) -> bool:
        """
        Delete index

        Returns:
        --------
        bool
            True if deleted
        """
        try:
            self.pc.delete_index(self.index_name)
            self.index = None
            return True
        except Exception as e:
            raise RuntimeError(f"Failed to delete collection: {str(e)}")

    def get_count(self) -> int:
        """
        Get number of vectors in index

        Returns:
        --------
        int
            Number of vectors
        """
        try:
            if self.index is None:
                return 0

            stats = self.index.describe_index_stats()
            return stats.total_vector_count if stats.total_vector_count else 0
        except Exception as e:
            # Index might not exist
            return 0
