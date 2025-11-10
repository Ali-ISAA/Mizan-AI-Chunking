"""
OpenAI embedding provider
"""

import time
from typing import List, Union, Optional

from .base import BaseEmbedder
from ..utils.config import get_config


class OpenAIEmbedder(BaseEmbedder):
    """OpenAI embedding implementation"""

    def __init__(self, model_name: str = "text-embedding-3-small",
                 dimension: int = 1536, api_key: Optional[str] = None,
                 base_url: Optional[str] = None):
        """
        Initialize OpenAI embedder

        Parameters:
        -----------
        model_name : str
            OpenAI embedding model name
        dimension : int
            Embedding dimension
        api_key : str, optional
            API key (loads from config if None)
        base_url : str, optional
            Base URL for API
        """
        super().__init__(model_name, dimension)

        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("openai package required. Install: pip install openai")

        # Get config
        config = get_config()
        self.api_key = api_key or config.openai_api_key
        self.base_url = base_url or config.openai_base_url

        if not self.api_key:
            raise ValueError("OpenAI API key not provided")

        # Initialize client
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )

    def embed(self, text: Union[str, List[str]]) -> Union[List[float], List[List[float]]]:
        """
        Generate embeddings for text(s)

        Parameters:
        -----------
        text : str or List[str]
            Text or list of texts to embed

        Returns:
        --------
        List[float] or List[List[float]]
            Single embedding or list of embeddings
        """
        if isinstance(text, str):
            response = self.client.embeddings.create(
                model=self.model_name,
                input=text,
                dimensions=self.dimension if self.model_name.startswith('text-embedding-3') else None
            )
            return response.data[0].embedding
        else:
            return self.embed_batch(text)

    def embed_batch(self, texts: List[str], batch_size: int = 100) -> List[List[float]]:
        """
        Generate embeddings for batch of texts

        Parameters:
        -----------
        texts : List[str]
            List of texts to embed
        batch_size : int
            Batch size for processing

        Returns:
        --------
        List[List[float]]
            List of embeddings
        """
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            print(f"  Embedding batch {i // batch_size + 1}/{(len(texts) - 1) // batch_size + 1}")

            # OpenAI API supports batch embedding
            response = self.client.embeddings.create(
                model=self.model_name,
                input=batch,
                dimensions=self.dimension if self.model_name.startswith('text-embedding-3') else None
            )

            # Extract embeddings
            embeddings = [data.embedding for data in response.data]
            all_embeddings.extend(embeddings)

            # Rate limiting
            time.sleep(0.1)

        return all_embeddings
