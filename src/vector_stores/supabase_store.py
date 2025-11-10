"""
Supabase vector store implementation
"""

from typing import List, Dict, Optional
import uuid
from supabase import create_client, Client

from .base import BaseVectorStore
from ..utils.config import get_config


class SupabaseStore(BaseVectorStore):
    """Supabase vector store implementation"""

    def __init__(self, collection_name: str, dimension: int):
        """
        Initialize Supabase store

        Parameters:
        -----------
        collection_name : str
            Name of the collection (table)
        dimension : int
            Dimension of embeddings
        """
        super().__init__(collection_name, dimension)

        # Load configuration
        config = get_config()

        if not config.supabase_url or not config.supabase_key:
            raise ValueError("Supabase URL and key are required")

        # Initialize Supabase client
        try:
            self.client: Client = create_client(
                config.supabase_url,
                config.supabase_key
            )
        except Exception as e:
            raise ConnectionError(f"Failed to initialize Supabase client: {str(e)}")

    def create_collection(self) -> bool:
        """
        Create collection (table) if it doesn't exist

        Returns:
        --------
        bool
            True if created, False if already exists
        """
        try:
            # Check if table exists by trying to query it
            try:
                self.client.table(self.collection_name).select("id").limit(1).execute()
                return False
            except Exception:
                # Table doesn't exist, create it via RPC or SQL
                # Note: This requires pgvector extension and proper table setup
                # Users should manually create the table with proper schema
                raise RuntimeError(
                    f"Table '{self.collection_name}' does not exist. "
                    f"Please create it manually with the following SQL:\n\n"
                    f"CREATE TABLE {self.collection_name} (\n"
                    f"  id TEXT PRIMARY KEY,\n"
                    f"  text TEXT NOT NULL,\n"
                    f"  embedding VECTOR({self.dimension}),\n"
                    f"  metadata JSONB,\n"
                    f"  created_at TIMESTAMPTZ DEFAULT NOW()\n"
                    f");\n\n"
                    f"CREATE INDEX ON {self.collection_name} USING ivfflat (embedding vector_cosine_ops);"
                )
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

            # Prepare records for insertion
            records = []
            for i in range(len(texts)):
                records.append({
                    'id': ids[i],
                    'text': texts[i],
                    'embedding': embeddings[i],
                    'metadata': metadata[i]
                })

            # Insert into Supabase
            self.client.table(self.collection_name).insert(records).execute()
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
            Metadata filters (JSONB query)

        Returns:
        --------
        List[Dict]
            List of results with text, metadata, and score
        """
        try:
            # Use RPC function for vector similarity search
            # Note: This requires a custom RPC function in Supabase
            # Users should create this function manually

            rpc_params = {
                'query_embedding': query_embedding,
                'match_count': top_k
            }

            if filters:
                rpc_params['filter'] = filters

            # Call RPC function
            response = self.client.rpc(
                f'match_{self.collection_name}',
                rpc_params
            ).execute()

            # Format results
            formatted_results = []
            for row in response.data:
                formatted_results.append({
                    'text': row.get('text', ''),
                    'metadata': row.get('metadata', {}),
                    'score': row.get('similarity', 0.0),
                    'id': row.get('id')
                })

            return formatted_results
        except Exception as e:
            # Fallback to basic query without similarity search
            try:
                query = self.client.table(self.collection_name).select("*").limit(top_k)

                if filters:
                    for key, value in filters.items():
                        query = query.eq(f"metadata->{key}", value)

                response = query.execute()

                formatted_results = []
                for row in response.data:
                    formatted_results.append({
                        'text': row.get('text', ''),
                        'metadata': row.get('metadata', {}),
                        'score': 0.0,
                        'id': row.get('id')
                    })

                return formatted_results
            except Exception as fallback_error:
                raise RuntimeError(f"Failed to search: {str(e)}. Fallback also failed: {str(fallback_error)}")

    def delete_collection(self) -> bool:
        """
        Delete collection (drop table)

        Returns:
        --------
        bool
            True if deleted
        """
        try:
            # Delete all records from table
            # Note: Actual table dropping requires SQL execution
            self.client.table(self.collection_name).delete().neq('id', '').execute()
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
            response = self.client.table(self.collection_name).select("id", count="exact").execute()
            return response.count if response.count is not None else 0
        except Exception as e:
            raise RuntimeError(f"Failed to get count: {str(e)}")
