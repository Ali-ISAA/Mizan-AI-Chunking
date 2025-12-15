"""
API Key Manager with automatic rotation, exponential backoff, and logging
"""

import time
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Callable, Any, Optional


class APIKeyManager:
    """
    Manages multiple API keys with automatic rotation on rate limit errors,
    exponential backoff, and file logging for debugging.
    """

    def __init__(self, api_keys: List[str], on_key_change: Optional[Callable[[str], None]] = None,
                 enable_logging: bool = True):
        """
        Initialize API key manager

        Parameters:
        -----------
        api_keys : List[str]
            List of API keys
        on_key_change : Callable[[str], None], optional
            Callback function called when key changes (for reconfiguring clients)
        enable_logging : bool
            Whether to log to file (default: True)
        """
        if not api_keys:
            raise ValueError("At least one API key is required")

        self.api_keys = list(api_keys)  # Make a copy
        self.current_index = 0
        self.rate_limited_keys = set()  # Temporary failures (rate limits)
        self.invalid_keys = set()  # Permanent failures (invalid keys)
        self.on_key_change = on_key_change

        # Backoff tracking
        self.consecutive_failures = 0
        self.base_delay = 2  # Base delay in seconds
        self.max_delay = 120  # Max delay in seconds (2 minutes)

        # Setup logging
        self.logger = None
        if enable_logging:
            self._setup_logging()

    def _setup_logging(self):
        """Setup file logging for API errors"""
        log_dir = Path(__file__).parent.parent.parent / "logs"
        log_dir.mkdir(exist_ok=True)

        log_file = log_dir / f"api_key_manager_{datetime.now().strftime('%Y%m%d')}.log"

        self.logger = logging.getLogger(f"APIKeyManager_{id(self)}")
        self.logger.setLevel(logging.DEBUG)

        # Avoid duplicate handlers
        if not self.logger.handlers:
            handler = logging.FileHandler(log_file)
            handler.setLevel(logging.DEBUG)
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

    def _log(self, level: str, message: str):
        """Log message to file"""
        if self.logger:
            getattr(self.logger, level.lower())(message)

    def _get_masked_key(self, key: str) -> str:
        """Return masked version of key for logging"""
        if len(key) > 8:
            return f"{key[:8]}...{key[-4:]}"
        return "***"

    def get_current_key(self) -> str:
        """Get current API key"""
        return self.api_keys[self.current_index]

    def get_valid_keys_count(self) -> int:
        """Get number of valid (non-invalid) keys"""
        return len(self.api_keys) - len(self.invalid_keys)

    def _calculate_backoff(self) -> float:
        """Calculate exponential backoff delay"""
        delay = min(self.base_delay * (2 ** self.consecutive_failures), self.max_delay)
        return delay

    def rotate_key(self) -> str:
        """
        Rotate to next available key

        Returns:
        --------
        str
            Next API key

        Raises:
        -------
        ValueError if no valid keys remain
        """
        # Check if we have any valid keys left
        if self.get_valid_keys_count() == 0:
            self._log('error', "No valid API keys remaining")
            raise ValueError("No valid API keys remaining. All keys are invalid.")

        old_index = self.current_index
        attempts = 0

        while attempts < len(self.api_keys):
            self.current_index = (self.current_index + 1) % len(self.api_keys)
            current_key = self.api_keys[self.current_index]

            # Skip permanently invalid keys
            if current_key in self.invalid_keys:
                attempts += 1
                continue

            # Skip temporarily rate-limited keys (unless all are rate-limited)
            if current_key not in self.rate_limited_keys:
                self._log('info', f"Rotated from key index {old_index} to {self.current_index}")
                # Notify callback
                if self.on_key_change:
                    self.on_key_change(current_key)
                return current_key

            attempts += 1

        # All valid keys are rate-limited - wait with backoff and clear rate limits
        backoff = self._calculate_backoff()
        self._log('warning', f"All keys rate-limited. Waiting {backoff}s before retry...")
        print(f"  ⏳ All API keys rate-limited. Waiting {backoff:.0f}s...")
        time.sleep(backoff)
        self.rate_limited_keys.clear()
        self.consecutive_failures += 1

        # Notify callback with current key after clearing
        current_key = self.get_current_key()
        if self.on_key_change:
            self.on_key_change(current_key)
        return current_key

    def mark_rate_limited(self):
        """Mark current key as temporarily rate-limited"""
        current_key = self.api_keys[self.current_index]
        self.rate_limited_keys.add(current_key)
        self._log('warning', f"Key {self._get_masked_key(current_key)} marked as rate-limited")

    def mark_invalid(self):
        """Mark current key as permanently invalid"""
        current_key = self.api_keys[self.current_index]
        self.invalid_keys.add(current_key)
        self._log('error', f"Key {self._get_masked_key(current_key)} marked as INVALID (permanent)")

    def execute_with_retry(
        self,
        func: Callable,
        *args,
        max_retries: Optional[int] = None,
        **kwargs
    ) -> Any:
        """
        Execute function with automatic key rotation and exponential backoff

        Parameters:
        -----------
        func : Callable
            Function to execute
        max_retries : int, optional
            Maximum number of retries (default: number of keys * 3)
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
            max_retries = len(self.api_keys) * 3  # Allow for multiple rounds

        last_exception = None

        for attempt in range(max_retries):
            # Check if we have valid keys
            if self.get_valid_keys_count() == 0:
                raise ValueError("No valid API keys remaining. All keys are invalid.")

            try:
                result = func(*args, **kwargs)

                # Success - reset backoff and clear rate-limited status
                self.consecutive_failures = 0
                current_key = self.api_keys[self.current_index]
                if current_key in self.rate_limited_keys:
                    self.rate_limited_keys.discard(current_key)

                return result

            except Exception as e:
                error_str = str(e).lower()
                self._log('debug', f"Attempt {attempt + 1}/{max_retries} failed: {str(e)[:200]}")

                # Check if it's an INVALID API KEY error (permanent failure)
                if any(term in error_str for term in ['api_key_invalid', 'api key not valid', 'invalid api key', 'invalid_api_key']):
                    self.mark_invalid()

                    if self.get_valid_keys_count() > 0:
                        self.rotate_key()
                        continue
                    else:
                        raise ValueError("No valid API keys remaining. All keys are invalid.")

                # Check if it's a rate limit error or timeout (temporary failure)
                elif any(term in error_str for term in ['429', 'quota', 'rate limit', 'rate_limit', 'resource_exhausted', 'resourceexhausted', 'timeout', 'timed out']):
                    self.mark_rate_limited()
                    self.consecutive_failures += 1

                    backoff = self._calculate_backoff()
                    self._log('warning', f"Rate limited. Backoff: {backoff}s, Attempt: {attempt + 1}/{max_retries}")

                    if attempt < max_retries - 1:
                        print(f"  ⏳ Rate limited. Waiting {backoff:.0f}s before retry...")
                        time.sleep(backoff)
                        self.rotate_key()
                        continue
                    else:
                        last_exception = e
                        break
                else:
                    # Unknown error - log and raise
                    self._log('error', f"Unknown error: {str(e)[:500]}")
                    raise e

        self._log('error', f"All retries exhausted. Last error: {str(last_exception)[:200]}")
        raise last_exception

    def get_status(self) -> dict:
        """Get manager status"""
        return {
            'total_keys': len(self.api_keys),
            'current_index': self.current_index,
            'invalid_keys': len(self.invalid_keys),
            'rate_limited_keys': len(self.rate_limited_keys),
            'valid_keys': self.get_valid_keys_count(),
            'available_keys': self.get_valid_keys_count() - len(self.rate_limited_keys),
            'consecutive_failures': self.consecutive_failures,
            'current_backoff': self._calculate_backoff()
        }
