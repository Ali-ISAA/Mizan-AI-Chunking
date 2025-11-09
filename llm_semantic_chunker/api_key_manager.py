"""
Google Gemini API Key Manager with Rotation
============================================

Manages multiple API keys and automatically rotates them when rate limits are hit.

FREE TIER LIMITS (per key):
---------------------------
Embedding API:
- RPM: 100 requests/minute
- TPM: 30,000 tokens/minute
- RPD: 1,000 requests/day

Text Generation (Gemini 2.5 Flash):
- RPM: 10 requests/minute
- TPM: 250,000 tokens/minute
- RPD: 250 requests/day
"""

import time
import google.generativeai as genai
from typing import List, Optional
import random

class APIKeyManager:
    """
    Manages multiple Gemini API keys with automatic rotation on rate limit errors.
    """

    def __init__(self, api_keys: List[str]):
        """
        Initialize the API key manager.

        Parameters:
        -----------
        api_keys : List[str]
            List of Google Gemini API keys
        """
        if not api_keys:
            raise ValueError("At least one API key is required")

        self.api_keys = api_keys
        self.current_key_index = 0
        self.failed_keys = set()  # Track temporarily failed keys
        self.key_usage = {key: {'requests': 0, 'last_reset': time.time()} for key in api_keys}

        print(f"[OK] Initialized with {len(api_keys)} API key(s)")
        self._configure_current_key()

    def _configure_current_key(self):
        """Configure genai to use the current API key."""
        current_key = self.api_keys[self.current_key_index]
        genai.configure(api_key=current_key)
        print(f"  Using API key #{self.current_key_index + 1} (ends with ...{current_key[-6:]})")

    def get_next_key(self):
        """
        Rotate to the next available API key.

        Returns:
        --------
        str
            The next API key
        """
        # Try all keys in rotation
        attempts = 0
        while attempts < len(self.api_keys):
            self.current_key_index = (self.current_key_index + 1) % len(self.api_keys)
            current_key = self.api_keys[self.current_key_index]

            # Skip temporarily failed keys
            if current_key in self.failed_keys:
                attempts += 1
                continue

            self._configure_current_key()
            return current_key

        # All keys failed - clear the failed set and retry
        print("[WARNING] All keys hit rate limits. Waiting 60 seconds before retry...")
        time.sleep(60)
        self.failed_keys.clear()
        return self.get_next_key()

    def mark_current_key_failed(self):
        """Mark the current key as temporarily failed."""
        current_key = self.api_keys[self.current_key_index]
        self.failed_keys.add(current_key)
        print(f"[WARNING] Key #{self.current_key_index + 1} hit rate limit. Rotating to next key...")

    def execute_with_retry(self, func, *args, max_retries: int = None, **kwargs):
        """
        Execute a function with automatic key rotation on rate limit errors.

        Parameters:
        -----------
        func : callable
            Function to execute (should use genai API)
        max_retries : int, optional
            Maximum number of key rotations to try (default: number of keys)
        *args, **kwargs
            Arguments to pass to the function

        Returns:
        --------
        Result of the function call

        Raises:
        -------
        Exception if all keys fail
        """
        if max_retries is None:
            max_retries = len(self.api_keys)

        last_exception = None

        for attempt in range(max_retries):
            try:
                # Execute the function
                result = func(*args, **kwargs)

                # Clear failed status on success
                current_key = self.api_keys[self.current_key_index]
                if current_key in self.failed_keys:
                    self.failed_keys.remove(current_key)

                return result

            except Exception as e:
                error_str = str(e)

                # Check if it's a rate limit error (429 or quota)
                if "429" in error_str or "quota" in error_str.lower() or "rate" in error_str.lower():
                    self.mark_current_key_failed()

                    if attempt < max_retries - 1:
                        # Rotate to next key
                        self.get_next_key()
                        print(f"  Retry attempt {attempt + 1}/{max_retries}...")
                        time.sleep(1)  # Brief pause before retry
                        continue
                    else:
                        last_exception = e
                        break
                else:
                    # Not a rate limit error - raise immediately
                    raise e

        # All retries exhausted
        print(f"\n[ERROR] All {max_retries} API keys exhausted")
        print("   Please wait for quota reset (midnight PST) or add more keys")
        print("   Check usage: https://ai.dev/usage")
        raise last_exception

    def get_current_key_info(self):
        """Get information about the current key."""
        return {
            'index': self.current_key_index,
            'key_suffix': self.api_keys[self.current_key_index][-6:],
            'total_keys': len(self.api_keys),
            'failed_keys': len(self.failed_keys),
            'available_keys': len(self.api_keys) - len(self.failed_keys)
        }


# Global instance - will be initialized when needed
_key_manager: Optional[APIKeyManager] = None


def initialize_key_manager(api_keys: List[str]):
    """
    Initialize the global API key manager.

    Parameters:
    -----------
    api_keys : List[str]
        List of Google Gemini API keys
    """
    global _key_manager
    # Only initialize if not already initialized
    if _key_manager is None:
        _key_manager = APIKeyManager(api_keys)
    return _key_manager


def get_key_manager(api_keys: Optional[List[str]] = None) -> APIKeyManager:
    """
    Get the global API key manager instance.
    If not initialized and api_keys are provided, initializes it automatically.

    Parameters:
    -----------
    api_keys : Optional[List[str]]
        API keys to use if key manager needs to be initialized

    Returns:
    --------
    APIKeyManager
        The global key manager instance
    """
    global _key_manager
    if _key_manager is None:
        if api_keys is not None:
            # Auto-initialize if keys are provided
            _key_manager = APIKeyManager(api_keys)
        else:
            raise RuntimeError("API key manager not initialized. Call initialize_key_manager() first or provide api_keys.")
    return _key_manager


def embed_with_retry(texts: List[str], model_name: str = "models/embedding-001") -> List[List[float]]:
    """
    Generate embeddings with automatic key rotation on rate limits.

    Parameters:
    -----------
    texts : List[str]
        List of texts to embed
    model_name : str
        Embedding model name

    Returns:
    --------
    List[List[float]]
        List of embedding vectors
    """
    manager = get_key_manager()

    def embed_func():
        embeddings = []
        for text in texts:
            # Truncate if too long
            if len(text) > 10000:
                text = text[:10000]

            result = genai.embed_content(
                model=model_name,
                content=text,
                task_type="retrieval_document"
            )
            embeddings.append(result['embedding'])

            # Add small delay to respect RPM limits (100 RPM = 0.6s per request)
            time.sleep(0.65)

        return embeddings

    return manager.execute_with_retry(embed_func)
