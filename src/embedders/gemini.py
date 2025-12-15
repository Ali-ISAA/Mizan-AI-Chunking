"""
Google Gemini embedding provider
"""

import time
import random
import sys
from typing import List, Union, Optional
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
import google.generativeai as genai

from .base import BaseEmbedder
from ..utils.api_key_manager import APIKeyManager
from ..utils.config import get_config

# Timeout for each embedding call (seconds)
EMBED_TIMEOUT = 30


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

        # Initialize key manager with callback to reconfigure genai
        self.key_manager = APIKeyManager(api_keys, on_key_change=self._reconfigure_client)

        # Configure with first key
        self._reconfigure_client(self.key_manager.get_current_key())

    def _reconfigure_client(self, api_key: str):
        """Reconfigure genai with new API key"""
        genai.configure(api_key=api_key)

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
        """Embed single text with timeout"""
        def embed_func():
            # Truncate if too long
            truncated_text = text[:10000] if len(text) > 10000 else text

            # Use ThreadPoolExecutor for timeout support
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(
                    genai.embed_content,
                    model=self.model_name,
                    content=truncated_text,
                    task_type="retrieval_document"
                )
                try:
                    result = future.result(timeout=EMBED_TIMEOUT)
                    return result['embedding']
                except FuturesTimeoutError:
                    raise TimeoutError(f"Embedding timed out after {EMBED_TIMEOUT}s - rate_limit")

        return self.key_manager.execute_with_retry(embed_func)

    def embed_batch(self, texts: List[str], batch_size: int = 100,
                    show_progress: bool = True) -> List[List[float]]:
        """
        Generate embeddings for batch of texts

        Parameters:
        -----------
        texts : List[str]
            List of texts to embed
        batch_size : int
            Batch size for processing
        show_progress : bool
            Show progress indicator for large batches

        Returns:
        --------
        List[List[float]]
            List of embeddings
        """
        all_embeddings = []
        total = len(texts)
        progress_interval = max(50, total // 20)  # Show progress every 50 or 5%

        for idx, text in enumerate(texts):
            embedding = self._embed_single(text)
            all_embeddings.append(embedding)

            # Show progress for large batches
            if show_progress and total > 100 and (idx + 1) % progress_interval == 0:
                pct = ((idx + 1) / total) * 100
                sys.stdout.write(f"\r    Embedding: {idx + 1}/{total} ({pct:.0f}%)")
                sys.stdout.flush()

            # Rate limiting with jitter to avoid thundering herd
            # Base: 1.0s (safe for 60 RPM per key with 2 keys = 120 RPM total)
            # Jitter: ±0.2s
            delay = 1.0 + random.uniform(-0.2, 0.2)
            time.sleep(delay)

        # Clear progress line
        if show_progress and total > 100:
            sys.stdout.write("\r" + " " * 50 + "\r")
            sys.stdout.flush()

        return all_embeddings
