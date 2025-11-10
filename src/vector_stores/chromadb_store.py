"""
ChromaDB vector store implementation
"""

from typing import List, Dict, Optional
import uuid
import chromadb
from chromadb.config import Settings

from .base import BaseVectorStore
from ..utils.config import get_config


class ChromaDBStore(BaseVectorStore):
    """ChromaDB vector store implementation"""

    def __init__(self, collection_name: str, dimension: int):
        """
        Initialize ChromaDB store

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

        # Initialize ChromaDB client
        try:
            # Check if using hosted ChromaDB
            if config.chromadb_api_key:
                self.client = chromadb.HttpClient(
                    host=config.chromadb_host,
                    port=config.chromadb_port,
                    settings=Settings(
                        chroma_client_auth_provider="chromadb.auth.token.TokenAuthClientProvider",
                        chroma_client_auth_credentials=config.chromadb_api_key
                    )
                )
            else:
                # Use local ChromaDB
                self.client = chromadb.PersistentClient(path="./chroma_db")

            self.collection = None
        except Exception as e:
            raise ConnectionError(f"Failed to initialize ChromaDB client: {str(e)}")

    def create_collection(self) -> bool:
        """
        Create collection if it doesn't exist

        Returns:
        --------
        bool
            True if created, False if already exists
        """
        try:
            # Try to get existing collection
            try:
                self.collection = self.client.get_collection(name=self.collection_name)
                return False
            except Exception:
                # Collection doesn't exist, create it
                self.collection = self.client.create_collection(
                    name=self.collection_name,
                    metadata={"dimension": self.dimension}
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
            if self.collection is None:
                self.create_collection()

            # Generate IDs if not provided
            if ids is None:
                ids = [str(uuid.uuid4()) for _ in range(len(texts))]

            # Prepare metadata
            if metadata is None:
                metadata = [{} for _ in range(len(texts))]

            # Insert into ChromaDB
            self.collection.add(
                embeddings=embeddings,
                documents=texts,
                metadatas=metadata,
                ids=ids
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
            if self.collection is None:
                raise ValueError("Collection not initialized. Call create_collection() first.")

            # Perform search
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                where=filters
            )

            # Format results
            formatted_results = []
            if results['documents'] and len(results['documents']) > 0:
                for i, doc in enumerate(results['documents'][0]):
                    formatted_results.append({
                        'text': doc,
                        'metadata': results['metadatas'][0][i] if results['metadatas'] else {},
                        'score': 1 - results['distances'][0][i] if results['distances'] else 0.0,  # Convert distance to similarity
                        'id': results['ids'][0][i] if results['ids'] else None
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
            self.client.delete_collection(name=self.collection_name)
            self.collection = None
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
            if self.collection is None:
                return 0
            return self.collection.count()
        except Exception as e:
            raise RuntimeError(f"Failed to get count: {str(e)}")
