"""
PostgreSQL pgvector store implementation
"""

from typing import List, Dict, Optional
import uuid
import psycopg2
from psycopg2.extras import execute_values
from psycopg2.extensions import register_adapter, AsIs
import numpy as np

from .base import BaseVectorStore
from ..utils.config import get_config


def adapt_numpy_array(array):
    """Adapter for numpy arrays to PostgreSQL arrays"""
    return AsIs(f"'[{','.join(map(str, array))}]'")

# Register numpy array adapter
register_adapter(np.ndarray, adapt_numpy_array)


class PgVectorStore(BaseVectorStore):
    """PostgreSQL pgvector store implementation"""

    def __init__(self, collection_name: str, dimension: int):
        """
        Initialize pgvector store

        Parameters:
        -----------
        collection_name : str
            Name of the table
        dimension : int
            Dimension of embeddings
        """
        super().__init__(collection_name, dimension)

        # Load configuration
        config = get_config()

        # Initialize PostgreSQL connection
        try:
            self.conn = psycopg2.connect(
                host=config.postgres_host,
                port=config.postgres_port,
                database=config.postgres_database,
                user=config.postgres_user,
                password=config.postgres_password
            )
            self.conn.autocommit = True

            # Enable pgvector extension
            with self.conn.cursor() as cur:
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        except Exception as e:
            raise ConnectionError(f"Failed to connect to PostgreSQL: {str(e)}")

    def create_collection(self) -> bool:
        """
        Create table if it doesn't exist

        Returns:
        --------
        bool
            True if created, False if already exists
        """
        try:
            with self.conn.cursor() as cur:
                # Check if table exists
                cur.execute("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables
                        WHERE table_name = %s
                    );
                """, (self.collection_name,))

                exists = cur.fetchone()[0]

                if exists:
                    return False

                # Create table
                cur.execute(f"""
                    CREATE TABLE {self.collection_name} (
                        id TEXT PRIMARY KEY,
                        text TEXT NOT NULL,
                        embedding VECTOR({self.dimension}),
                        metadata JSONB,
                        created_at TIMESTAMPTZ DEFAULT NOW()
                    );
                """)

                # Create index for similarity search
                cur.execute(f"""
                    CREATE INDEX ON {self.collection_name}
                    USING ivfflat (embedding vector_cosine_ops)
                    WITH (lists = 100);
                """)

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

            # Prepare records
            records = []
            for i in range(len(texts)):
                # Convert embedding to string format for PostgreSQL
                embedding_str = '[' + ','.join(map(str, embeddings[i])) + ']'
                records.append((
                    ids[i],
                    texts[i],
                    embedding_str,
                    psycopg2.extras.Json(metadata[i])
                ))

            # Insert into PostgreSQL
            with self.conn.cursor() as cur:
                execute_values(
                    cur,
                    f"""
                    INSERT INTO {self.collection_name} (id, text, embedding, metadata)
                    VALUES %s
                    ON CONFLICT (id) DO UPDATE SET
                        text = EXCLUDED.text,
                        embedding = EXCLUDED.embedding,
                        metadata = EXCLUDED.metadata
                    """,
                    records
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
            Metadata filters (JSONB query)

        Returns:
        --------
        List[Dict]
            List of results with text, metadata, and score
        """
        try:
            # Convert embedding to string format
            embedding_str = '[' + ','.join(map(str, query_embedding)) + ']'

            # Build query
            query = f"""
                SELECT id, text, metadata,
                       1 - (embedding <=> %s::vector) AS similarity
                FROM {self.collection_name}
            """

            params = [embedding_str]

            # Add filters if provided
            if filters:
                where_clauses = []
                for key, value in filters.items():
                    where_clauses.append(f"metadata->>%s = %s")
                    params.extend([key, str(value)])

                if where_clauses:
                    query += " WHERE " + " AND ".join(where_clauses)

            query += " ORDER BY embedding <=> %s::vector LIMIT %s"
            params.extend([embedding_str, top_k])

            # Execute search
            with self.conn.cursor() as cur:
                cur.execute(query, params)
                rows = cur.fetchall()

            # Format results
            formatted_results = []
            for row in rows:
                formatted_results.append({
                    'id': row[0],
                    'text': row[1],
                    'metadata': row[2] if row[2] else {},
                    'score': float(row[3]) if row[3] else 0.0
                })

            return formatted_results
        except Exception as e:
            raise RuntimeError(f"Failed to search: {str(e)}")

    def delete_collection(self) -> bool:
        """
        Delete table

        Returns:
        --------
        bool
            True if deleted
        """
        try:
            with self.conn.cursor() as cur:
                cur.execute(f"DROP TABLE IF EXISTS {self.collection_name};")
            return True
        except Exception as e:
            raise RuntimeError(f"Failed to delete collection: {str(e)}")

    def get_count(self) -> int:
        """
        Get number of vectors in table

        Returns:
        --------
        int
            Number of vectors
        """
        try:
            with self.conn.cursor() as cur:
                cur.execute(f"SELECT COUNT(*) FROM {self.collection_name};")
                count = cur.fetchone()[0]
            return count
        except Exception as e:
            # Table might not exist
            return 0

    def __del__(self):
        """Close connection on deletion"""
        if hasattr(self, 'conn') and self.conn:
            self.conn.close()
