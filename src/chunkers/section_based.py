"""
Section-based chunker - splits only at markdown headers
"""

import re
from typing import List, Dict, Optional

from .base import BaseChunker


class SectionBasedChunker(BaseChunker):
    """Chunks text by markdown sections only (no size limits)"""

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 0):
        """
        Initialize section-based chunker

        Note: chunk_size is ignored, sections are kept whole
        """
        super().__init__(chunk_size, chunk_overlap)

    def chunk(self, text: str, metadata: Optional[Dict] = None) -> List[Dict]:
        """
        Chunk text by markdown sections

        Parameters:
        -----------
        text : str
            Text to chunk
        metadata : Dict, optional
            Additional metadata

        Returns:
        --------
        List[Dict]
            List of chunks (one per section)
        """
        sections = self._extract_sections(text)

        chunks = []
        for i, section in enumerate(sections):
            chunk_dict = self._create_chunk_dict(section['content'], i, metadata)
            chunk_dict['metadata']['section_header'] = section['header']
            chunk_dict['metadata']['section_level'] = section['level']
            chunks.append(chunk_dict)

        return chunks

    def _extract_sections(self, text: str) -> List[Dict]:
        """Extract markdown sections"""
        sections = []
        pattern = r'^(#{1,3})\s+(.+?)$'

        # Split by headers
        parts = re.split(pattern, text, flags=re.MULTILINE)

        if len(parts) == 1:
            # No headers
            return [{'header': 'Document', 'level': 0, 'content': text}]

        # First part
        if parts[0].strip():
            sections.append({
                'header': 'Introduction',
                'level': 0,
                'content': parts[0].strip()
            })

        # Process headers
        for i in range(1, len(parts), 3):
            if i + 2 < len(parts):
                level = len(parts[i])
                header = parts[i + 1].strip()
                content = parts[i + 2].strip()

                # Include header in content
                full_content = f"{'#' * level} {header}\n\n{content}"

                sections.append({
                    'header': header,
                    'level': level,
                    'content': full_content
                })

        return sections
