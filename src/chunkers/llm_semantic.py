"""
LLM Semantic chunker - uses LLM to determine semantic boundaries
"""

import re
import json
from typing import List, Dict, Optional

from .base import BaseChunker
from ..llms import get_llm
from ..utils.config import get_config


class LLMSemanticChunker(BaseChunker):
    """Uses LLM to intelligently determine chunk boundaries based on semantics"""

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 0,
                 llm_provider: Optional[str] = None, llm_model: Optional[str] = None):
        """
        Initialize LLM semantic chunker

        Parameters:
        -----------
        chunk_size : int
            Target chunk size (used for post-processing only)
        chunk_overlap : int
            Overlap between chunks
        llm_provider : str, optional
            LLM provider (loads from config if None)
        llm_model : str, optional
            LLM model name (loads from config if None)
        """
        super().__init__(chunk_size, chunk_overlap)

        # Get config
        config = get_config()
        self.llm_provider = llm_provider or config.llm_provider
        self.llm_model = llm_model or config.llm_model

        # Initialize LLM
        self.llm = get_llm(self.llm_provider, self.llm_model, temperature=0.2)

    def chunk(self, text: str, metadata: Optional[Dict] = None) -> List[Dict]:
        """
        Chunk text using LLM semantic analysis

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
        print(f"Using LLM semantic chunking with {self.llm_provider}/{self.llm_model}")

        # Split into sections if markdown
        sections = self._extract_sections(text)

        all_chunks = []
        chunk_index = 0

        for section in sections:
            section_text = section['content']
            section_tokens = self.count_tokens(section_text)

            print(f"  Processing section: {section['header'][:50]}... ({section_tokens} tokens)")

            # If section is small enough, keep as is
            if section_tokens <= self.chunk_size:
                chunk_dict = self._create_chunk_dict(section_text, chunk_index, metadata)
                chunk_dict['metadata']['section_header'] = section['header']
                all_chunks.append(chunk_dict)
                chunk_index += 1
            else:
                # Use LLM to split large section
                section_chunks = self._llm_split_section(section_text, section['header'])

                for chunk_text in section_chunks:
                    chunk_dict = self._create_chunk_dict(chunk_text, chunk_index, metadata)
                    chunk_dict['metadata']['section_header'] = section['header']
                    all_chunks.append(chunk_dict)
                    chunk_index += 1

        # Filter out empty chunks
        filtered_chunks = [c for c in all_chunks if c['text'].strip() and c['tokens'] > 0]

        # Re-index chunks after filtering
        for i, chunk in enumerate(filtered_chunks):
            chunk['metadata']['chunk_index'] = i

        if len(filtered_chunks) < len(all_chunks):
            print(f"  Filtered out {len(all_chunks) - len(filtered_chunks)} empty chunks")

        print(f"Created {len(filtered_chunks)} semantic chunks")
        return filtered_chunks

    def _extract_sections(self, text: str) -> List[Dict]:
        """Extract markdown sections or treat as single section"""
        # Check if markdown
        if re.search(r'^#{1,3}\s+', text, re.MULTILINE):
            sections = []
            pattern = r'^(#{1,3})\s+(.+?)$'

            # Split by headers
            parts = re.split(pattern, text, flags=re.MULTILINE)

            if len(parts) == 1:
                # No headers found
                return [{'header': 'Document', 'level': 0, 'content': text}]

            # First part (before any header)
            if parts[0].strip():
                sections.append({
                    'header': 'Introduction',
                    'level': 0,
                    'content': parts[0].strip()
                })

            # Process header groups
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
        else:
            # Not markdown, treat as single section
            return [{'header': 'Document', 'level': 0, 'content': text}]

    def _llm_split_section(self, section_text: str, section_header: str) -> List[str]:
        """Use LLM to split section into semantic chunks"""
        # Create prompt for LLM
        system_prompt = """You are a document chunking expert. Your task is to identify natural breakpoints in the text where topics or themes change.

Analyze the text and determine where to split it into semantically coherent chunks. Each chunk should contain related information about a single sub-topic.

Return ONLY a JSON array of split points (character indices where splits should occur). For example:
[150, 430, 680]

This means split at character 150, 430, and 680."""

        user_prompt = f"""Analyze this text section and determine optimal split points for semantic chunking.

Section: {section_header}

Text:
{section_text[:4000]}

Return only the JSON array of split indices."""

        try:
            # Get LLM response
            response = self.llm.generate_with_retry(user_prompt, system_prompt)

            # Parse split points
            split_points = self._parse_split_points(response)

            if split_points:
                # Split text at these points
                chunks = []
                start = 0

                for split_point in split_points:
                    if split_point > start and split_point < len(section_text):
                        chunk = section_text[start:split_point].strip()
                        if chunk:
                            chunks.append(chunk)
                        start = split_point

                # Add final chunk
                if start < len(section_text):
                    chunk = section_text[start:].strip()
                    if chunk:
                        chunks.append(chunk)

                return chunks if chunks else [section_text]
            else:
                # LLM didn't provide split points, fall back to simple split
                return self._fallback_split(section_text)

        except Exception as e:
            print(f"    LLM chunking failed: {e}, using fallback")
            return self._fallback_split(section_text)

    def _parse_split_points(self, response: str) -> List[int]:
        """Parse split points from LLM response"""
        try:
            # Try to find JSON array in response
            match = re.search(r'\[[\d,\s]+\]', response)
            if match:
                return json.loads(match.group())
            return []
        except:
            return []

    def _fallback_split(self, text: str) -> List[str]:
        """Fallback: split by paragraphs respecting chunk size"""
        paragraphs = text.split('\n\n')
        chunks = []
        current_chunk = []
        current_tokens = 0

        for para in paragraphs:
            para_tokens = self.count_tokens(para)

            if current_tokens + para_tokens > self.chunk_size and current_chunk:
                chunks.append('\n\n'.join(current_chunk))
                current_chunk = [para]
                current_tokens = para_tokens
            else:
                current_chunk.append(para)
                current_tokens += para_tokens

        if current_chunk:
            chunks.append('\n\n'.join(current_chunk))

        return chunks if chunks else [text]
