"""
Ollama embedding provider for local embeddings
"""

import time
from typing import List, Union, Optional

from .base import BaseEmbedder
from ..utils.config import get_config


class OllamaEmbedder(BaseEmbedder):
    """Ollama embedding implementation"""

    def __init__(self, model_name: str = "nomic-embed-text",
                 dimension: int = 768, base_url: Optional[str] = None):
        """
        Initialize Ollama embedder

        Parameters:
        -----------
        model_name : str
            Ollama embedding model name
        dimension : int
            Embedding dimension
        base_url : str, optional
            Ollama server URL (default: http://localhost:11434)
        """
        super().__init__(model_name, dimension)

        try:
            import ollama
        except ImportError:
            raise ImportError("ollama package required. Install: pip install ollama")

        # Get config
        config = get_config()
        self.base_url = base_url or config.ollama_base_url

        # Initialize client
        self.client = ollama.Client(host=self.base_url)

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
            response = self.client.embeddings(
                model=self.model_name,
                prompt=text
            )
            return response['embedding']
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

            # Ollama doesn't support batch embedding, so we embed one at a time
            for text in batch:
                response = self.client.embeddings(
                    model=self.model_name,
                    prompt=text
                )
                all_embeddings.append(response['embedding'])

                # Small delay for local models
                time.sleep(0.01)

        return all_embeddings
