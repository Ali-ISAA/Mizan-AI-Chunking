"""
Configuration loader for environment variables
"""

import os
from pathlib import Path
from typing import Optional, Dict, List
from dotenv import load_dotenv


class Config:
    """Manages configuration from environment variables"""

    def __init__(self, env_file: Optional[str] = None):
        """
        Initialize configuration

        Parameters:
        -----------
        env_file : str, optional
            Path to .env file. If None, looks for .env in project root
        """
        if env_file:
            env_path = Path(env_file)
        else:
            env_path = Path(__file__).parent.parent.parent / ".env"

        load_dotenv(dotenv_path=env_path)
        self._validate_config()

    def _validate_config(self):
        """Validate that required configuration is present"""
        # Check for at least one LLM provider
        if not (self.get_gemini_keys() or self.openai_api_key or
                self.ollama_base_url or self.litellm_api_key):
            raise ValueError(
                "No LLM provider configured! Set at least one of: "
                "GEMINI_API_KEY_*, OPENAI_API_KEY, OLLAMA_BASE_URL, or LITELLM_API_KEY"
            )

    # ==================== LLM Configuration ====================

    @property
    def llm_provider(self) -> str:
        """Get LLM provider (gemini, openai, ollama, litellm)"""
        return os.getenv('LLM_PROVIDER', 'gemini').lower()

    @property
    def llm_model(self) -> str:
        """Get LLM model name"""
        default_models = {
            'gemini': 'gemini-2.0-flash-lite',
            'openai': 'gpt-4o-mini',
            'ollama': 'llama3.2',
            'litellm': 'gpt-4o-mini'
        }
        return os.getenv('LLM_MODEL', default_models.get(self.llm_provider, 'gpt-4o-mini'))

    def get_gemini_keys(self) -> List[str]:
        """Get all Gemini API keys for rotation"""
        keys = []
        for i in range(1, 11):  # Support up to 10 keys
            key = os.getenv(f'GEMINI_API_KEY_{i}')
            if key:
                keys.append(key)
        return keys

    @property
    def openai_api_key(self) -> Optional[str]:
        """Get OpenAI API key"""
        return os.getenv('OPENAI_API_KEY')

    @property
    def openai_base_url(self) -> Optional[str]:
        """Get OpenAI base URL (for vLLM, etc.)"""
        return os.getenv('OPENAI_BASE_URL')

    @property
    def ollama_base_url(self) -> str:
        """Get Ollama base URL"""
        return os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')

    @property
    def litellm_api_key(self) -> Optional[str]:
        """Get LiteLLM API key"""
        return os.getenv('LITELLM_API_KEY')

    # ==================== Embedding Configuration ====================

    @property
    def embedding_provider(self) -> str:
        """Get embedding provider (gemini, openai, ollama)"""
        return os.getenv('EMBEDDING_PROVIDER', 'gemini').lower()

    @property
    def embedding_model(self) -> str:
        """Get embedding model name"""
        default_models = {
            'gemini': 'models/embedding-001',
            'openai': 'text-embedding-3-small',
            'ollama': 'nomic-embed-text'
        }
        return os.getenv('EMBEDDING_MODEL', default_models.get(self.embedding_provider, 'text-embedding-3-small'))

    @property
    def embedding_dimension(self) -> int:
        """Get embedding dimension"""
        default_dims = {
            'gemini': 768,
            'openai': 1536,
            'ollama': 768
        }
        return int(os.getenv('EMBEDDING_DIMENSION',
                            default_dims.get(self.embedding_provider, 768)))

    # ==================== Vector Store Configuration ====================

    @property
    def vector_store(self) -> str:
        """Get vector store type (chromadb, supabase, pgvector, qdrant, weaviate, pinecone)"""
        return os.getenv('VECTOR_STORE', 'chromadb').lower()

    @property
    def collection_name(self) -> str:
        """Get collection/table name"""
        return os.getenv('COLLECTION_NAME', 'documents')

    # ChromaDB
    @property
    def chromadb_api_key(self) -> Optional[str]:
        return os.getenv('CHROMADB_API_KEY')

    @property
    def chromadb_tenant(self) -> Optional[str]:
        return os.getenv('CHROMADB_TENANT')

    @property
    def chromadb_database(self) -> Optional[str]:
        return os.getenv('CHROMADB_DATABASE')

    @property
    def chromadb_host(self) -> str:
        return os.getenv('CHROMADB_HOST', 'localhost')

    @property
    def chromadb_port(self) -> int:
        return int(os.getenv('CHROMADB_PORT', 8000))

    # Supabase
    @property
    def supabase_url(self) -> Optional[str]:
        return os.getenv('SUPABASE_URL')

    @property
    def supabase_key(self) -> Optional[str]:
        return os.getenv('SUPABASE_SERVICE_KEY') or os.getenv('SUPABASE_ANON_KEY')

    # PostgreSQL (pgvector)
    @property
    def postgres_host(self) -> str:
        return os.getenv('POSTGRES_HOST', 'localhost')

    @property
    def postgres_port(self) -> int:
        return int(os.getenv('POSTGRES_PORT', 5432))

    @property
    def postgres_database(self) -> str:
        return os.getenv('POSTGRES_DATABASE', 'vectordb')

    @property
    def postgres_user(self) -> str:
        return os.getenv('POSTGRES_USER', 'postgres')

    @property
    def postgres_password(self) -> Optional[str]:
        return os.getenv('POSTGRES_PASSWORD')

    # Qdrant
    @property
    def qdrant_url(self) -> str:
        return os.getenv('QDRANT_URL', 'http://localhost:6333')

    @property
    def qdrant_api_key(self) -> Optional[str]:
        return os.getenv('QDRANT_API_KEY')

    # Weaviate
    @property
    def weaviate_url(self) -> str:
        return os.getenv('WEAVIATE_URL', 'http://localhost:8080')

    @property
    def weaviate_api_key(self) -> Optional[str]:
        return os.getenv('WEAVIATE_API_KEY')

    # Pinecone
    @property
    def pinecone_api_key(self) -> Optional[str]:
        return os.getenv('PINECONE_API_KEY')

    @property
    def pinecone_environment(self) -> Optional[str]:
        return os.getenv('PINECONE_ENVIRONMENT')

    @property
    def pinecone_index(self) -> str:
        return os.getenv('PINECONE_INDEX', 'documents')

    # ==================== Chunking Configuration ====================

    @property
    def chunk_size(self) -> int:
        """Default chunk size in tokens"""
        return int(os.getenv('CHUNK_SIZE', 512))

    @property
    def chunk_overlap(self) -> int:
        """Default chunk overlap in tokens"""
        return int(os.getenv('CHUNK_OVERLAP', 50))

    def get_all_config(self) -> Dict:
        """Get all configuration as dictionary"""
        return {
            'llm_provider': self.llm_provider,
            'llm_model': self.llm_model,
            'embedding_provider': self.embedding_provider,
            'embedding_model': self.embedding_model,
            'vector_store': self.vector_store,
            'collection_name': self.collection_name,
            'chunk_size': self.chunk_size,
            'chunk_overlap': self.chunk_overlap
        }


# Global config instance
_config: Optional[Config] = None


def get_config(env_file: Optional[str] = None) -> Config:
    """
    Get or create global config instance

    Parameters:
    -----------
    env_file : str, optional
        Path to .env file

    Returns:
    --------
    Config
        Configuration instance
    """
    global _config
    if _config is None:
        _config = Config(env_file)
    return _config
