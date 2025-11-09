"""
Markdown Section-Based Chunker
===============================

Chunks markdown files based on their section structure (headers).
Keeps complete knowledge together by preserving entire sections.

Features:
- Chunks by ## (H2) and ### (H3) headers
- Preserves parent section context in child sections
- Handles markdown structure intelligently
- No fixed token limits - respects natural document structure
"""

import re
from typing import List, Dict, Tuple
from pathlib import Path


class MarkdownSectionChunker:
    """
    Chunks markdown files by sections, preserving document structure.
    """

    def __init__(self, md_file_path: str):
        """
        Initialize the chunker with a markdown file.

        Parameters:
        -----------
        md_file_path : str
            Path to the markdown (.md) file
        """
        self.md_file_path = md_file_path
        self.file_name = Path(md_file_path).stem
        self.content = self._read_file()

    def _read_file(self) -> str:
        """Read the markdown file content."""
        with open(self.md_file_path, 'r', encoding='utf-8') as f:
            return f.read()

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

        # Split by headers (## or ###)
        # Pattern matches: ## or ### followed by any text
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

    def chunk_by_sections(self) -> Tuple[List[str], List[Dict[str, str]]]:
        """
        Chunk the markdown file by sections.

        Returns:
        --------
        Tuple[List[str], List[Dict]]
            - List of chunk texts
            - List of chunk metadata (title, level, parent)
        """
        sections = self._extract_sections()

        chunks = []
        metadata = []

        for section in sections:
            # Build chunk text with context
            chunk_parts = []

            # Add parent context if this is a subsection
            if section['parent']:
                chunk_parts.append(f"[Parent Section: {section['parent']}]")

            # Add section title
            if section['level'] == 2:
                chunk_parts.append(f"## {section['title']}")
            elif section['level'] == 3:
                chunk_parts.append(f"### {section['title']}")

            # Add section content
            chunk_parts.append(section['content'])

            # Combine and clean
            chunk_text = '\n\n'.join(chunk_parts)
            chunk_text = self._clean_content(chunk_text)

            # Only add if chunk has substantial content (more than 50 chars)
            if len(chunk_text) >= 50:
                chunks.append(chunk_text)
                metadata.append({
                    'title': section['title'],
                    'level': section['level'],
                    'parent': section['parent'],
                    'length': len(chunk_text),
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
            output_path = f"output_markdown_section_chunks.txt"

        chunks, metadata = self.chunk_by_sections()

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("MARKDOWN SECTION-BASED CHUNKING RESULTS\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Source File: {self.md_file_path}\n")
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
                f.write(f"Length: {meta['length']} characters\n")
                f.write(f"{'-'*80}\n")
                f.write(chunk)
                f.write(f"\n{'-'*80}\n")

        print(f"\n✓ Saved {len(chunks)} chunks to: {output_path}")
        print(f"\n📊 Chunking Statistics:")
        print(f"  - Total Chunks: {len(chunks)}")
        print(f"  - H2 Sections: {sum(1 for m in metadata if m['level'] == 2)}")
        print(f"  - H3 Subsections: {sum(1 for m in metadata if m['level'] == 3)}")
        print(f"  - Average Length: {sum(m['length'] for m in metadata) // len(metadata)} chars")
        print(f"  - Min Length: {min(m['length'] for m in metadata)} chars")
        print(f"  - Max Length: {max(m['length'] for m in metadata)} chars")


def main():
    """
    Main function to demonstrate markdown section chunking.
    """
    print("=" * 80)
    print("MARKDOWN SECTION-BASED CHUNKER")
    print("=" * 80)
    print("\nThis chunker preserves complete knowledge by section structure.")
    print("It chunks based on markdown headers (## and ###).\n")

    # Path to the markdown file
    md_file_path = r"C:\Personal\Mizan AI\Chunking\Digital Government Policies - V2.0.pdf_processed.md"

    print(f"📄 Processing: {md_file_path}\n")

    # Create chunker
    chunker = MarkdownSectionChunker(md_file_path)

    # Get chunks
    print("🔄 Extracting sections and creating chunks...\n")
    chunks, metadata = chunker.chunk_by_sections()

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
        print(f"Length: {meta['length']} chars")
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
