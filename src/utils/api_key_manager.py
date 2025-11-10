"""
API Key Manager with automatic rotation for rate limit handling
"""

import time
from typing import List, Callable, Any, Optional


class APIKeyManager:
    """
    Manages multiple API keys with automatic rotation on rate limit errors
    """

    def __init__(self, api_keys: List[str]):
        """
        Initialize API key manager

        Parameters:
        -----------
        api_keys : List[str]
            List of API keys
        """
        if not api_keys:
            raise ValueError("At least one API key is required")

        self.api_keys = api_keys
        self.current_index = 0
        self.failed_keys = set()

    def get_current_key(self) -> str:
        """Get current API key"""
        return self.api_keys[self.current_index]

    def rotate_key(self) -> str:
        """
        Rotate to next available key

        Returns:
        --------
        str
            Next API key
        """
        attempts = 0
        while attempts < len(self.api_keys):
            self.current_index = (self.current_index + 1) % len(self.api_keys)
            current_key = self.api_keys[self.current_index]

            if current_key not in self.failed_keys:
                return current_key

            attempts += 1

        # All keys failed - clear and retry
        print("[WARNING] All API keys hit rate limits. Waiting 60 seconds...")
        time.sleep(60)
        self.failed_keys.clear()
        return self.get_current_key()

    def mark_current_failed(self):
        """Mark current key as failed"""
        current_key = self.api_keys[self.current_index]
        self.failed_keys.add(current_key)

    def execute_with_retry(
        self,
        func: Callable,
        *args,
        max_retries: Optional[int] = None,
        **kwargs
    ) -> Any:
        """
        Execute function with automatic key rotation on rate limit errors

        Parameters:
        -----------
        func : Callable
            Function to execute
        max_retries : int, optional
            Maximum number of retries (default: number of keys)
        *args, **kwargs
            Arguments for the function

        Returns:
        --------
        Any
            Function result

        Raises:
        -------
        Exception if all retries fail
        """
        if max_retries is None:
            max_retries = len(self.api_keys)

        last_exception = None

        for attempt in range(max_retries):
            try:
                result = func(*args, **kwargs)

                # Clear failed status on success
                current_key = self.api_keys[self.current_index]
                if current_key in self.failed_keys:
                    self.failed_keys.remove(current_key)

                return result

            except Exception as e:
                error_str = str(e).lower()

                # Check if it's a rate limit error
                if any(term in error_str for term in ['429', 'quota', 'rate limit', 'rate_limit']):
                    self.mark_current_failed()

                    if attempt < max_retries - 1:
                        self.rotate_key()
                        print(f"  Rotating to next API key (attempt {attempt + 1}/{max_retries})")
                        time.sleep(1)
                        continue
                    else:
                        last_exception = e
                        break
                else:
                    # Not a rate limit error - raise immediately
                    raise e

        print(f"\n[ERROR] All {max_retries} API keys exhausted")
        raise last_exception

    def get_status(self) -> dict:
        """Get manager status"""
        return {
            'total_keys': len(self.api_keys),
            'current_index': self.current_index,
            'failed_keys': len(self.failed_keys),
            'available_keys': len(self.api_keys) - len(self.failed_keys)
        }
