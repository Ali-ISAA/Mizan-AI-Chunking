"""
Recursive chunker - splits text recursively by separators
"""

import re
from typing import List, Dict, Optional

from .base import BaseChunker


class RecursiveChunker(BaseChunker):
    """Recursively chunks text by trying different separators"""

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50,
                 separators: Optional[List[str]] = None):
        """
        Initialize recursive chunker

        Parameters:
        -----------
        chunk_size : int
            Target chunk size in tokens
        chunk_overlap : int
            Overlap between chunks
        separators : List[str], optional
            List of separators to try (default: paragraph, newline, sentence, word)
        """
        super().__init__(chunk_size, chunk_overlap)

        if separators is None:
            # Default separators in order of preference
            self.separators = [
                "\n\n",  # Paragraphs
                "\n",    # Lines
                ". ",    # Sentences
                "! ",    # Sentences
                "? ",    # Sentences
                " ",     # Words
                ""       # Characters
            ]
        else:
            self.separators = separators

    def chunk(self, text: str, metadata: Optional[Dict] = None) -> List[Dict]:
        """
        Chunk text recursively

        Parameters:
        -----------
        text : str
            Text to chunk
        metadata : Dict, optional
            Additional metadata

        Returns:
        --------
        List[Dict]
            List of chunks
        """
        raw_chunks = self._recursive_split(text, self.separators)

        # Convert to chunk dicts
        chunks = []
        for i, chunk_text in enumerate(raw_chunks):
            chunks.append(self._create_chunk_dict(chunk_text, i, metadata))

        return chunks

    def _recursive_split(self, text: str, separators: List[str]) -> List[str]:
        """Recursively split text"""
        final_chunks = []
        separator = separators[-1]

        # Try each separator
        for i, sep in enumerate(separators):
            if sep == "":
                separator = sep
                break
            if re.search(re.escape(sep), text):
                separator = sep
                new_separators = separators[i + 1:]
                break

        # Split by separator
        splits = self._split_text(text, separator)

        # Merge splits into chunks
        merged = self._merge_splits(splits, separator)

        # Process each merged chunk
        for chunk in merged:
            if self.count_tokens(chunk) > self.chunk_size:
                # Still too large, recurse
                if len(separators) > 1:
                    final_chunks.extend(self._recursive_split(chunk, separators[1:]))
                else:
                    final_chunks.append(chunk)
            else:
                final_chunks.append(chunk)

        return final_chunks

    def _split_text(self, text: str, separator: str) -> List[str]:
        """Split text by separator"""
        if separator:
            splits = re.split(f"({re.escape(separator)})", text)
            # Rejoin separator with following text
            result = []
            for i in range(0, len(splits) - 1, 2):
                if i + 1 < len(splits):
                    result.append(splits[i] + splits[i + 1])
                else:
                    result.append(splits[i])
            if len(splits) % 2 == 1:
                result.append(splits[-1])
            return [s for s in result if s]
        else:
            return list(text)

    def _merge_splits(self, splits: List[str], separator: str) -> List[str]:
        """Merge splits into chunks respecting size limits"""
        chunks = []
        current_chunk = []
        current_tokens = 0

        for split in splits:
            split_tokens = self.count_tokens(split)
            sep_tokens = self.count_tokens(separator) if separator else 0

            if current_tokens + split_tokens + sep_tokens > self.chunk_size:
                if current_chunk:
                    chunks.append(separator.join(current_chunk))

                    # Handle overlap
                    while current_tokens > self.chunk_overlap and len(current_chunk) > 1:
                        removed = current_chunk.pop(0)
                        current_tokens -= self.count_tokens(removed) + sep_tokens

                current_chunk.append(split)
                current_tokens = split_tokens
            else:
                current_chunk.append(split)
                current_tokens += split_tokens + (sep_tokens if current_chunk else 0)

        if current_chunk:
            chunks.append(separator.join(current_chunk))

        return chunks
