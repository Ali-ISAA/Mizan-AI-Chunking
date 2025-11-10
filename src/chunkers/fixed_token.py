"""
Fixed token chunker - splits text into equal-sized token chunks
"""

from typing import List, Dict, Optional

from .base import BaseChunker


class FixedTokenChunker(BaseChunker):
    """Chunks text into fixed-size token chunks"""

    def chunk(self, text: str, metadata: Optional[Dict] = None) -> List[Dict]:
        """
        Chunk text into fixed-size pieces

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
        chunks = []
        start_pos = 0
        chunk_index = 0

        while start_pos < len(text):
            # Estimate character length for target tokens (rough: 4 chars per token)
            estimated_end = start_pos + (self.chunk_size * 4)

            # Get candidate chunk
            candidate = text[start_pos:min(estimated_end, len(text))]
            candidate_tokens = self.count_tokens(candidate)

            # Adjust to hit target token count
            if candidate_tokens < self.chunk_size and estimated_end < len(text):
                # Need more characters
                while candidate_tokens < self.chunk_size and len(candidate) < len(text) - start_pos:
                    estimated_end += 100
                    candidate = text[start_pos:min(estimated_end, len(text))]
                    candidate_tokens = self.count_tokens(candidate)

            elif candidate_tokens > self.chunk_size:
                # Need fewer characters
                while candidate_tokens > self.chunk_size and len(candidate) > 100:
                    estimated_end -= 100
                    candidate = text[start_pos:estimated_end]
                    candidate_tokens = self.count_tokens(candidate)

            # Snap to word boundary
            if estimated_end < len(text):
                last_space = candidate.rfind(' ')
                last_newline = candidate.rfind('\n')
                snap_pos = max(last_space, last_newline)

                if snap_pos > len(candidate) // 2:
                    candidate = candidate[:snap_pos + 1]
                    estimated_end = start_pos + snap_pos + 1

            chunks.append(self._create_chunk_dict(candidate, chunk_index, metadata))
            chunk_index += 1

            # Calculate next start with overlap
            if self.chunk_overlap > 0 and estimated_end < len(text):
                overlap_chars = min(self.chunk_overlap * 4, len(candidate))
                overlap_start = max(start_pos, estimated_end - overlap_chars)

                # Fine-tune overlap
                overlap_text = text[overlap_start:estimated_end]
                overlap_tokens = self.count_tokens(overlap_text)

                while overlap_tokens > self.chunk_overlap and overlap_start < estimated_end - 10:
                    overlap_start += 10
                    overlap_text = text[overlap_start:estimated_end]
                    overlap_tokens = self.count_tokens(overlap_text)

                start_pos = overlap_start
            else:
                start_pos = estimated_end

        return chunks
