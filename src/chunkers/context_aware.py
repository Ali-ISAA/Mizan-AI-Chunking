"""
Context-aware chunker - preserves section context from markdown headers
"""

import re
from typing import List, Dict, Optional

from .base import BaseChunker


class ContextAwareChunker(BaseChunker):
    """Chunks text while preserving markdown section context"""

    def chunk(self, text: str, metadata: Optional[Dict] = None) -> List[Dict]:
        """
        Chunk text with context awareness

        Parameters:
        -----------
        text : str
            Text to chunk
        metadata : Dict, optional
            Additional metadata

        Returns:
        --------
        List[Dict]
            List of chunks with preserved context
        """
        # Extract sections
        sections = self._extract_sections(text)

        chunks = []
        chunk_index = 0

        for section in sections:
            section_text = section['content']
            section_tokens = self.count_tokens(section_text)

            # Small sections: keep whole
            if section_tokens <= self.chunk_size:
                chunk_dict = self._create_chunk_dict(section_text, chunk_index, metadata)
                chunk_dict['metadata']['section_header'] = section['header']
                chunk_dict['metadata']['section_level'] = section['level']
                chunks.append(chunk_dict)
                chunk_index += 1
            else:
                # Large sections: split intelligently
                section_chunks = self._split_large_section(
                    section_text,
                    section['header']
                )

                for chunk_text in section_chunks:
                    chunk_dict = self._create_chunk_dict(chunk_text, chunk_index, metadata)
                    chunk_dict['metadata']['section_header'] = section['header']
                    chunk_dict['metadata']['section_level'] = section['level']
                    chunks.append(chunk_dict)
                    chunk_index += 1

        # Filter out empty chunks
        filtered_chunks = [c for c in chunks if c['text'].strip() and c['tokens'] > 0]

        # Re-index chunks after filtering
        for i, chunk in enumerate(filtered_chunks):
            chunk['metadata']['chunk_index'] = i

        return filtered_chunks

    def _extract_sections(self, text: str) -> List[Dict]:
        """Extract markdown sections"""
        sections = []
        pattern = r'^(#{1,3})\s+(.+?)$'

        # Split by headers
        parts = re.split(pattern, text, flags=re.MULTILINE)

        if len(parts) == 1:
            # No headers
            return [{'header': '', 'level': 0, 'content': text}]

        # First part
        if parts[0].strip():
            sections.append({
                'header': '',
                'level': 0,
                'content': parts[0].strip()
            })

        # Process headers
        for i in range(1, len(parts), 3):
            if i + 2 < len(parts):
                level = len(parts[i])
                header = parts[i + 1].strip()
                content = parts[i + 2].strip()

                sections.append({
                    'header': header,
                    'level': level,
                    'content': content
                })

        return sections

    def _split_large_section(self, text: str, header: str) -> List[str]:
        """Split large section at paragraph boundaries"""
        paragraphs = text.split('\n\n')
        chunks = []
        current_chunk = []
        current_tokens = 0

        # Include header context in first chunk
        header_context = f"# {header}\n\n" if header else ""
        header_tokens = self.count_tokens(header_context)

        for para in paragraphs:
            para_tokens = self.count_tokens(para)
            total_tokens = current_tokens + para_tokens + (header_tokens if not current_chunk else 0)

            if total_tokens > self.chunk_size and current_chunk:
                # Create chunk with header context
                chunk_text = header_context + '\n\n'.join(current_chunk) if header_context and not chunks else '\n\n'.join(current_chunk)
                chunks.append(chunk_text)

                # Start new chunk with overlap
                if self.chunk_overlap > 0 and len(current_chunk) > 1:
                    current_chunk = [current_chunk[-1], para]
                    current_tokens = self.count_tokens(current_chunk[-2]) + para_tokens
                else:
                    current_chunk = [para]
                    current_tokens = para_tokens
            else:
                current_chunk.append(para)
                current_tokens += para_tokens

        if current_chunk:
            chunk_text = header_context + '\n\n'.join(current_chunk) if header_context and not chunks else '\n\n'.join(current_chunk)
            chunks.append(chunk_text)

        return chunks if chunks else [text]
