"""
Google Gemini embedding provider
"""

import time
from typing import List, Union, Optional
import google.generativeai as genai

from .base import BaseEmbedder
from ..utils.api_key_manager import APIKeyManager
from ..utils.config import get_config


class GeminiEmbedder(BaseEmbedder):
    """Google Gemini embedding implementation"""

    def __init__(self, model_name: str = "models/embedding-001",
                 dimension: int = 768, api_keys: Optional[List[str]] = None):
        """
        Initialize Gemini embedder

        Parameters:
        -----------
        model_name : str
            Gemini embedding model name
        dimension : int
            Embedding dimension
        api_keys : List[str], optional
            List of API keys for rotation (loads from config if None)
        """
        super().__init__(model_name, dimension)

        # Get API keys
        if api_keys is None:
            config = get_config()
            api_keys = config.get_gemini_keys()

        if not api_keys:
            raise ValueError("No Gemini API keys provided")

        # Initialize key manager
        self.key_manager = APIKeyManager(api_keys)

        # Configure with first key
        genai.configure(api_key=self.key_manager.get_current_key())

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
            return self._embed_single(text)
        else:
            return self.embed_batch(text)

    def _embed_single(self, text: str) -> List[float]:
        """Embed single text"""
        def embed_func():
            # Truncate if too long
            truncated_text = text[:10000] if len(text) > 10000 else text

            result = genai.embed_content(
                model=self.model_name,
                content=truncated_text,
                task_type="retrieval_document"
            )
            return result['embedding']

        return self.key_manager.execute_with_retry(embed_func)

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

            for text in batch:
                embedding = self._embed_single(text)
                all_embeddings.append(embedding)

                # Rate limiting: 100 RPM = 0.6s per request
                time.sleep(0.65)

        return all_embeddings
