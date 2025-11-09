"""
Context-Aware Markdown Chunker
===============================

Intelligent chunker that combines markdown section structure with context-aware splitting.

Features:
- Uses markdown headers (## ###) as primary boundaries
- Splits large sections intelligently while preserving context
- Keeps parent section titles for context
- Maintains semantic coherence
- Handles both small and large sections appropriately

Strategy:
1. Extract sections based on markdown headers
2. For small sections (<= max_tokens): keep as single chunk
3. For large sections (> max_tokens): split intelligently at:
   - Paragraph boundaries
   - Sentence boundaries
   - While maintaining parent context
"""

import re
import tiktoken
from typing import List, Dict, Tuple
from pathlib import Path


class ContextAwareMarkdownChunker:
    """
    Intelligent markdown chunker that preserves context while handling large sections.
    """

    def __init__(self, md_file_path: str, max_chunk_tokens: int = 512, overlap_tokens: int = 50):
        """
        Initialize the context-aware markdown chunker.

        Parameters:
        -----------
        md_file_path : str
            Path to the markdown (.md) file
        max_chunk_tokens : int
            Maximum tokens per chunk (default: 512)
        overlap_tokens : int
            Token overlap between chunks for context (default: 50)
        """
        self.md_file_path = md_file_path
        self.file_name = Path(md_file_path).stem
        self.max_chunk_tokens = max_chunk_tokens
        self.overlap_tokens = overlap_tokens
        self.encoding = tiktoken.get_encoding("cl100k_base")
        self.content = self._read_file()

    def _read_file(self) -> str:
        """Read the markdown file content."""
        with open(self.md_file_path, 'r', encoding='utf-8') as f:
            return f.read()

    def _count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        return len(self.encoding.encode(text))

    def _clean_content(self, text: str) -> str:
        """
        Clean content by removing image placeholders and extra whitespace.

        Parameters:
        -----------
        text : str
            Raw text content

        Returns:
        --------
        str
            Cleaned text
        """
        # Remove image placeholders
        text = re.sub(r'<!-- image -->\s*', '', text)

        # Remove multiple consecutive newlines (more than 2)
        text = re.sub(r'\n{3,}', '\n\n', text)

        # Remove leading/trailing whitespace
        text = text.strip()

        return text

    def _extract_sections(self) -> List[Dict[str, str]]:
        """
        Extract sections from markdown based on headers.

        Returns:
        --------
        List[Dict]
            List of sections with metadata
        """
        sections = []
        lines = self.content.split('\n')

        current_section = {
            'level': 0,
            'title': 'Document Header',
            'content': [],
            'parent': None
        }

        parent_h2 = None  # Track parent H2 section

        for line in lines:
            # Check for H2 header (##)
            h2_match = re.match(r'^##\s+(.+)$', line)
            if h2_match:
                # Save previous section if it has content
                if current_section['content']:
                    section_text = '\n'.join(current_section['content']).strip()
                    if section_text:
                        current_section['content'] = section_text
                        sections.append(current_section.copy())

                # Start new H2 section
                title = h2_match.group(1).strip()
                parent_h2 = title
                current_section = {
                    'level': 2,
                    'title': title,
                    'content': [],
                    'parent': None
                }
                continue

            # Check for H3 header (###)
            h3_match = re.match(r'^###\s+(.+)$', line)
            if h3_match:
                # Save previous section if it has content
                if current_section['content']:
                    section_text = '\n'.join(current_section['content']).strip()
                    if section_text:
                        current_section['content'] = section_text
                        sections.append(current_section.copy())

                # Start new H3 section (child of current H2)
                title = h3_match.group(1).strip()
                current_section = {
                    'level': 3,
                    'title': title,
                    'content': [],
                    'parent': parent_h2
                }
                continue

            # Add line to current section content
            current_section['content'].append(line)

        # Don't forget the last section
        if current_section['content']:
            section_text = '\n'.join(current_section['content']).strip()
            if section_text:
                current_section['content'] = section_text
                sections.append(current_section.copy())

        return sections

    def _split_large_section(self, section: Dict, context_header: str) -> List[Tuple[str, Dict]]:
        """
        Split a large section into smaller chunks while preserving context.

        Parameters:
        -----------
        section : Dict
            Section dictionary with content
        context_header : str
            Header with parent context to prepend

        Returns:
        --------
        List[Tuple[str, Dict]]
            List of (chunk_text, metadata) tuples
        """
        chunks = []
        content = section['content']

        # Split content into paragraphs (double newline)
        paragraphs = re.split(r'\n\n+', content)

        current_chunk = []
        current_tokens = self._count_tokens(context_header)

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            para_tokens = self._count_tokens(para)

            # If single paragraph exceeds max, split by sentences
            if para_tokens > self.max_chunk_tokens:
                # Save current chunk if not empty
                if current_chunk:
                    chunk_text = context_header + '\n\n' + '\n\n'.join(current_chunk)
                    chunk_text = self._clean_content(chunk_text)
                    chunks.append((chunk_text, {
                        'title': section['title'],
                        'level': section['level'],
                        'parent': section['parent'],
                        'chunk_part': len(chunks) + 1,
                        'is_split': True
                    }))
                    current_chunk = []
                    current_tokens = self._count_tokens(context_header)

                # Split paragraph by sentences
                sentences = re.split(r'(?<=[.!?])\s+', para)
                sentence_chunk = []
                sentence_tokens = self._count_tokens(context_header)

                for sentence in sentences:
                    sent_tokens = self._count_tokens(sentence)

                    if sentence_tokens + sent_tokens > self.max_chunk_tokens:
                        if sentence_chunk:
                            chunk_text = context_header + '\n\n' + ' '.join(sentence_chunk)
                            chunk_text = self._clean_content(chunk_text)
                            chunks.append((chunk_text, {
                                'title': section['title'],
                                'level': section['level'],
                                'parent': section['parent'],
                                'chunk_part': len(chunks) + 1,
                                'is_split': True
                            }))
                        sentence_chunk = [sentence]
                        sentence_tokens = self._count_tokens(context_header + '\n\n' + sentence)
                    else:
                        sentence_chunk.append(sentence)
                        sentence_tokens += sent_tokens

                # Add remaining sentences
                if sentence_chunk:
                    chunk_text = context_header + '\n\n' + ' '.join(sentence_chunk)
                    chunk_text = self._clean_content(chunk_text)
                    chunks.append((chunk_text, {
                        'title': section['title'],
                        'level': section['level'],
                        'parent': section['parent'],
                        'chunk_part': len(chunks) + 1,
                        'is_split': True
                    }))

            # If adding this paragraph would exceed max, save current chunk
            elif current_tokens + para_tokens > self.max_chunk_tokens:
                if current_chunk:
                    chunk_text = context_header + '\n\n' + '\n\n'.join(current_chunk)
                    chunk_text = self._clean_content(chunk_text)
                    chunks.append((chunk_text, {
                        'title': section['title'],
                        'level': section['level'],
                        'parent': section['parent'],
                        'chunk_part': len(chunks) + 1,
                        'is_split': True
                    }))

                # Start new chunk with overlap (last paragraph)
                if current_chunk:
                    current_chunk = [current_chunk[-1], para]
                    current_tokens = self._count_tokens(context_header + '\n\n' + '\n\n'.join(current_chunk))
                else:
                    current_chunk = [para]
                    current_tokens = self._count_tokens(context_header + '\n\n' + para)
            else:
                current_chunk.append(para)
                current_tokens += para_tokens

        # Add remaining chunk
        if current_chunk:
            chunk_text = context_header + '\n\n' + '\n\n'.join(current_chunk)
            chunk_text = self._clean_content(chunk_text)
            chunks.append((chunk_text, {
                'title': section['title'],
                'level': section['level'],
                'parent': section['parent'],
                'chunk_part': len(chunks) + 1,
                'is_split': True
            }))

        return chunks

    def chunk_by_context_aware_sections(self) -> Tuple[List[str], List[Dict[str, str]]]:
        """
        Chunk the markdown file using context-aware section splitting.

        Returns:
        --------
        Tuple[List[str], List[Dict]]
            - List of chunk texts
            - List of chunk metadata
        """
        sections = self._extract_sections()

        chunks = []
        metadata = []

        for section in sections:
            # Build context header
            context_parts = []

            # Add parent context if this is a subsection
            if section['parent']:
                context_parts.append(f"[Parent Section: {section['parent']}]")

            # Add section title
            if section['level'] == 2:
                context_parts.append(f"## {section['title']}")
            elif section['level'] == 3:
                context_parts.append(f"### {section['title']}")

            context_header = '\n'.join(context_parts)

            # Count tokens in section content
            content_tokens = self._count_tokens(section['content'])
            header_tokens = self._count_tokens(context_header)

            # If section fits in one chunk, keep it intact
            if header_tokens + content_tokens <= self.max_chunk_tokens:
                chunk_text = context_header + '\n\n' + section['content']
                chunk_text = self._clean_content(chunk_text)

                if len(chunk_text) >= 50:  # Min 50 chars
                    chunks.append(chunk_text)
                    metadata.append({
                        'title': section['title'],
                        'level': section['level'],
                        'parent': section['parent'],
                        'tokens': self._count_tokens(chunk_text),
                        'file': self.file_name,
                        'is_split': False
                    })
            else:
                # Section is too large, split it intelligently
                split_chunks = self._split_large_section(section, context_header)

                for chunk_text, meta in split_chunks:
                    if len(chunk_text) >= 50:  # Min 50 chars
                        chunks.append(chunk_text)
                        metadata.append({
                            **meta,
                            'tokens': self._count_tokens(chunk_text),
                            'file': self.file_name
                        })

        return chunks, metadata

    def save_chunks_to_file(self, output_path: str = None):
        """
        Save chunks to a text file for review.

        Parameters:
        -----------
        output_path : str, optional
            Path to save the output file. If None, uses default naming.
        """
        if output_path is None:
            output_path = f"output_context_aware_md_chunks.txt"

        chunks, metadata = self.chunk_by_context_aware_sections()

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("CONTEXT-AWARE MARKDOWN CHUNKING RESULTS\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Source File: {self.md_file_path}\n")
            f.write(f"Max Chunk Tokens: {self.max_chunk_tokens}\n")
            f.write(f"Total Chunks: {len(chunks)}\n")
            f.write("=" * 80 + "\n\n")

            for i, (chunk, meta) in enumerate(zip(chunks, metadata), 1):
                f.write(f"\n{'='*80}\n")
                f.write(f"CHUNK #{i}\n")
                f.write(f"{'='*80}\n")
                f.write(f"Title: {meta['title']}\n")
                f.write(f"Level: H{meta['level']}\n")
                if meta['parent']:
                    f.write(f"Parent Section: {meta['parent']}\n")
                f.write(f"Tokens: {meta['tokens']}\n")
                f.write(f"Split: {'Yes (Part ' + str(meta.get('chunk_part', 1)) + ')' if meta['is_split'] else 'No'}\n")
                f.write(f"{'-'*80}\n")
                f.write(chunk)
                f.write(f"\n{'-'*80}\n")

        print(f"\n✓ Saved {len(chunks)} chunks to: {output_path}")
        print(f"\n📊 Chunking Statistics:")
        print(f"  - Total Chunks: {len(chunks)}")
        print(f"  - H2 Sections: {sum(1 for m in metadata if m['level'] == 2)}")
        print(f"  - H3 Subsections: {sum(1 for m in metadata if m['level'] == 3)}")
        print(f"  - Split Chunks: {sum(1 for m in metadata if m['is_split'])}")
        print(f"  - Intact Sections: {sum(1 for m in metadata if not m['is_split'])}")
        print(f"  - Average Tokens: {sum(m['tokens'] for m in metadata) // len(metadata)}")
        print(f"  - Min Tokens: {min(m['tokens'] for m in metadata)}")
        print(f"  - Max Tokens: {max(m['tokens'] for m in metadata)}")


def main():
    """
    Main function to demonstrate context-aware markdown chunking.
    """
    print("=" * 80)
    print("CONTEXT-AWARE MARKDOWN CHUNKER")
    print("=" * 80)
    print("\nIntelligent chunker that:")
    print("  • Uses markdown structure as primary boundaries")
    print("  • Splits large sections while preserving context")
    print("  • Keeps small sections intact")
    print("  • Maintains parent section information\n")

    # Path to the markdown file
    md_file_path = r"C:\Personal\Mizan AI\Chunking\Digital Government Policies - V2.0.pdf_processed.md"

    print(f"📄 Processing: {md_file_path}\n")

    # Create chunker
    chunker = ContextAwareMarkdownChunker(
        md_file_path=md_file_path,
        max_chunk_tokens=512,
        overlap_tokens=50
    )

    # Get chunks
    print("🔄 Extracting sections and creating context-aware chunks...\n")
    chunks, metadata = chunker.chunk_by_context_aware_sections()

    print(f"✓ Created {len(chunks)} chunks\n")

    # Preview first 3 chunks
    print("=" * 80)
    print("PREVIEW: First 3 Chunks")
    print("=" * 80)

    for i, (chunk, meta) in enumerate(zip(chunks[:3], metadata[:3]), 1):
        print(f"\n--- Chunk #{i} ---")
        print(f"Title: {meta['title']}")
        print(f"Level: H{meta['level']}")
        if meta['parent']:
            print(f"Parent: {meta['parent']}")
        print(f"Tokens: {meta['tokens']}")
        print(f"Split: {'Yes (Part ' + str(meta.get('chunk_part', 1)) + ')' if meta['is_split'] else 'No'}")
        print(f"\nContent Preview (first 300 chars):")
        print(chunk[:300] + "..." if len(chunk) > 300 else chunk)
        print("-" * 80)

    # Save all chunks to file
    print("\n" + "=" * 80)
    print("Saving all chunks to file...")
    print("=" * 80)
    chunker.save_chunks_to_file()


if __name__ == "__main__":
    main()
