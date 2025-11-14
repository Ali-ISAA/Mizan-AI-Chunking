"""Vector store implementations with lazy loading"""

from .base import BaseVectorStore

__all__ = ['BaseVectorStore', 'get_vector_store']


def get_vector_store(store_type: str, collection_name: str, dimension: int, **kwargs) -> BaseVectorStore:
    """
    Factory function to get vector store instance

    Uses lazy imports to avoid loading unnecessary dependencies.
    Only the requested vector store is imported and initialized.

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
        from .chromadb_store import ChromaDBStore
        return ChromaDBStore(collection_name=collection_name, dimension=dimension, **kwargs)
    elif store_type == 'supabase':
        from .supabase_store import SupabaseStore
        return SupabaseStore(collection_name=collection_name, dimension=dimension, **kwargs)
    elif store_type == 'pgvector':
        from .pgvector_store import PgVectorStore
        return PgVectorStore(collection_name=collection_name, dimension=dimension, **kwargs)
    elif store_type == 'qdrant':
        from .qdrant_store import QdrantStore
        return QdrantStore(collection_name=collection_name, dimension=dimension, **kwargs)
    elif store_type == 'weaviate':
        from .weaviate_store import WeaviateStore
        return WeaviateStore(collection_name=collection_name, dimension=dimension, **kwargs)
    elif store_type == 'pinecone':
        from .pinecone_store import PineconeStore
        return PineconeStore(collection_name=collection_name, dimension=dimension, **kwargs)
    else:
        raise ValueError(f"Unknown vector store type: {store_type}")
