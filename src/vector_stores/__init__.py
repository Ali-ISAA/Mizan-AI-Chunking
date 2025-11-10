"""Vector store implementations"""

from .base import BaseVectorStore
from .chromadb_store import ChromaDBStore
from .supabase_store import SupabaseStore
from .pgvector_store import PgVectorStore
from .qdrant_store import QdrantStore
from .weaviate_store import WeaviateStore
from .pinecone_store import PineconeStore

__all__ = [
    'BaseVectorStore',
    'ChromaDBStore',
    'SupabaseStore',
    'PgVectorStore',
    'QdrantStore',
    'WeaviateStore',
    'PineconeStore'
]


def get_vector_store(store_type: str, collection_name: str, dimension: int, **kwargs) -> BaseVectorStore:
    """
    Factory function to get vector store instance

    Parameters:
    -----------
    store_type : str
        Type of vector store (chromadb, supabase, pgvector, qdrant, weaviate, pinecone)
    collection_name : str
        Name of collection/table/index
    dimension : int
        Embedding dimension
    **kwargs
        Additional arguments for the store

    Returns:
    --------
    BaseVectorStore
        Vector store instance
    """
    store_type = store_type.lower()

    if store_type == 'chromadb':
        return ChromaDBStore(collection_name=collection_name, dimension=dimension, **kwargs)
    elif store_type == 'supabase':
        return SupabaseStore(collection_name=collection_name, dimension=dimension, **kwargs)
    elif store_type == 'pgvector':
        return PgVectorStore(collection_name=collection_name, dimension=dimension, **kwargs)
    elif store_type == 'qdrant':
        return QdrantStore(collection_name=collection_name, dimension=dimension, **kwargs)
    elif store_type == 'weaviate':
        return WeaviateStore(collection_name=collection_name, dimension=dimension, **kwargs)
    elif store_type == 'pinecone':
        return PineconeStore(collection_name=collection_name, dimension=dimension, **kwargs)
    else:
        raise ValueError(f"Unknown vector store type: {store_type}")
