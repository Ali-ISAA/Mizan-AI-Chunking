"""
Context-Aware PDF Chunker
==========================

Intelligent chunker for PDF files that extracts text and chunks based on document structure.

Features:
- Extracts text from PDFs with OCR support for images
- Detects document structure (headings, sections)
- Splits large sections intelligently while preserving context
- Handles both text-based and image-based PDFs
- Maintains semantic coherence

Strategy:
1. Extract text from PDF (with OCR if needed)
2. Detect section boundaries using text patterns
3. For small sections (<= max_tokens): keep as single chunk
4. For large sections (> max_tokens): split intelligently at:
   - Paragraph boundaries
   - Sentence boundaries
   - While maintaining context
"""

import re
import tiktoken
import pdfplumber
from typing import List, Dict, Tuple
from pathlib import Path
import numpy as np

# OCR support (optional)
try:
    import easyocr
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False


class ContextAwarePDFChunker:
    """
    Intelligent PDF chunker that preserves context while handling large sections.
    """

    def __init__(self, pdf_file_path: str, max_chunk_tokens: int = 512, overlap_tokens: int = 50, use_ocr: bool = True):
        """
        Initialize the context-aware PDF chunker.

        Parameters:
        -----------
        pdf_file_path : str
            Path to the PDF file
        max_chunk_tokens : int
            Maximum tokens per chunk (default: 512)
        overlap_tokens : int
            Token overlap between chunks for context (default: 50)
        use_ocr : bool
            Whether to use OCR for images (default: True)
        """
        self.pdf_file_path = pdf_file_path
        self.file_name = Path(pdf_file_path).stem
        self.max_chunk_tokens = max_chunk_tokens
        self.overlap_tokens = overlap_tokens
        self.use_ocr = use_ocr and OCR_AVAILABLE
        self.encoding = tiktoken.get_encoding("cl100k_base")

        # Initialize OCR if available and requested
        self.ocr_reader = None
        if self.use_ocr:
            print("🔄 Initializing EasyOCR (English + Arabic)...")
            self.ocr_reader = easyocr.Reader(['en', 'ar'], gpu=False)
            print("✓ EasyOCR initialized")

        # Extract text from PDF
        self.content = self._extract_text_from_pdf()

    def _count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        return len(self.encoding.encode(text))

    def _has_images(self, page) -> bool:
        """Check if page has meaningful images (not just logos)."""
        if not hasattr(page, 'images') or not page.images:
            return False

        for img in page.images:
            width = abs(img['x1'] - img['x0'])
            height = abs(img['y1'] - img['y0'])
            # Filter out small images (logos, icons)
            if width > 50 and height > 50:
                return True
        return False

    def _extract_text_from_image(self, page, page_num) -> str:
        """Extract text from images in page using OCR."""
        if not self.ocr_reader:
            return ""

        try:
            # Convert page to image (150 DPI for faster processing)
            page_image = page.to_image(resolution=150)
            pil_image = page_image.original

            # Convert PIL Image to numpy array (required by EasyOCR)
            image_np = np.array(pil_image)

            # Perform OCR on the entire page image
            ocr_results = self.ocr_reader.readtext(image_np)

            if ocr_results:
                # Extract text from OCR results
                ocr_texts = []
                for detection in ocr_results:
                    text = detection[1]  # detection is (bbox, text, confidence)
                    confidence = detection[2]
                    if confidence > 0.3:  # Only include if confidence > 30%
                        ocr_texts.append(text)

                if ocr_texts:
                    ocr_combined = ' '.join(ocr_texts)
                    print(f"   ✓ Extracted {len(ocr_texts)} text segments from page {page_num}")
                    return ocr_combined

            return ""
        except Exception as e:
            print(f"⚠️  OCR failed for page {page_num}: {e}")
            return ""

    def _fix_text_spacing(self, text: str) -> str:
        """Fix spacing issues in extracted text."""
        # Pattern 1: Lowercase letter followed by uppercase letter
        text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
        # Pattern 2: Digit followed by uppercase letter
        text = re.sub(r'(\d)([A-Z][a-z])', r'\1 \2', text)
        # Pattern 3: Lowercase followed by digit
        text = re.sub(r'([a-z])(\d+)([A-Z\s])', r'\1 \2\3', text)
        # Pattern 4: Punctuation without space before uppercase
        text = re.sub(r'([.!?])([A-Z])', r'\1 \2', text)
        return text

    def _extract_text_from_pdf(self) -> str:
        """Extract text from PDF with OCR support."""
        print(f"\n📄 Extracting text from PDF: {self.pdf_file_path}")

        full_text = []

        with pdfplumber.open(self.pdf_file_path) as pdf:
            total_pages = len(pdf.pages)
            print(f"📖 Total pages: {total_pages}\n")

            for page_num, page in enumerate(pdf.pages, 1):
                page_text = ""

                # Extract regular text
                extracted = page.extract_text(x_tolerance=2, y_tolerance=3)
                if extracted:
                    page_text = extracted

                # Check if page has images for OCR
                if self.use_ocr and self._has_images(page):
                    print(f"  🖼️  Page {page_num}: Extracting text from images...")
                    ocr_text = self._extract_text_from_image(page, page_num)
                    if ocr_text:
                        page_text += "\n" + ocr_text

                # Fix spacing issues
                if page_text:
                    page_text = self._fix_text_spacing(page_text)
                    full_text.append(f"--- Page {page_num} ---\n{page_text}")

                if page_num % 10 == 0:
                    print(f"  ✓ Processed {page_num}/{total_pages} pages")

        print(f"\n✓ Extracted text from {total_pages} pages")

        return "\n\n".join(full_text)

    def _clean_content(self, text: str) -> str:
        """Clean content by removing extra whitespace and page markers."""
        # Remove page markers for cleaner chunks
        text = re.sub(r'--- Page \d+ ---\s*', '', text)

        # Remove multiple consecutive newlines (more than 2)
        text = re.sub(r'\n{3,}', '\n\n', text)

        # Remove leading/trailing whitespace
        text = text.strip()

        return text

    def _detect_sections(self) -> List[Dict[str, str]]:
        """
        Detect sections in the PDF text using patterns.

        Looks for:
        - Numbered sections (01., 02., 1., 2., etc.)
        - ALL CAPS headings
        - Section keywords

        Returns:
        --------
        List[Dict]
            List of sections with metadata
        """
        sections = []

        # Split by page markers first
        pages = re.split(r'--- Page \d+ ---', self.content)

        current_section = {
            'title': 'Document Start',
            'content': [],
            'page_start': 1
        }

        current_page = 1

        for page_content in pages:
            if not page_content.strip():
                current_page += 1
                continue

            lines = page_content.split('\n')

            for line in lines:
                line_stripped = line.strip()

                # Pattern 1: Numbered sections (e.g., "01. Title", "1. Title", "07. Title")
                section_match = re.match(r'^(\d{1,2})\s*[.\)]\s*(.+)$', line_stripped)

                # Pattern 2: ALL CAPS headings (minimum 3 words)
                caps_match = re.match(r'^([A-Z][A-Z\s]{10,})$', line_stripped)

                if section_match and len(section_match.group(2)) > 5:
                    # Save previous section
                    if current_section['content']:
                        section_text = '\n'.join(current_section['content']).strip()
                        if section_text and len(section_text) > 50:
                            current_section['content'] = section_text
                            sections.append(current_section.copy())

                    # Start new section
                    title = section_match.group(2).strip()
                    current_section = {
                        'title': f"{section_match.group(1)}. {title}",
                        'content': [],
                        'page_start': current_page
                    }

                elif caps_match and len(caps_match.group(1).split()) >= 2:
                    # Save previous section
                    if current_section['content']:
                        section_text = '\n'.join(current_section['content']).strip()
                        if section_text and len(section_text) > 50:
                            current_section['content'] = section_text
                            sections.append(current_section.copy())

                    # Start new section with caps title
                    current_section = {
                        'title': caps_match.group(1).strip(),
                        'content': [],
                        'page_start': current_page
                    }

                else:
                    # Add to current section
                    if line_stripped:
                        current_section['content'].append(line)

            current_page += 1

        # Don't forget last section
        if current_section['content']:
            section_text = '\n'.join(current_section['content']).strip()
            if section_text and len(section_text) > 50:
                current_section['content'] = section_text
                sections.append(current_section.copy())

        return sections

    def _split_large_section(self, section: Dict, context_header: str) -> List[Tuple[str, Dict]]:
        """
        Split a large section into smaller chunks while preserving context.

        Same logic as markdown chunker but for PDF sections.
        """
        chunks = []
        content = section['content']

        # Split content into paragraphs
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
                        'page_start': section['page_start'],
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
                                'page_start': section['page_start'],
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
                        'page_start': section['page_start'],
                        'chunk_part': len(chunks) + 1,
                        'is_split': True
                    }))

            # If adding paragraph exceeds max, save current chunk
            elif current_tokens + para_tokens > self.max_chunk_tokens:
                if current_chunk:
                    chunk_text = context_header + '\n\n' + '\n\n'.join(current_chunk)
                    chunk_text = self._clean_content(chunk_text)
                    chunks.append((chunk_text, {
                        'title': section['title'],
                        'page_start': section['page_start'],
                        'chunk_part': len(chunks) + 1,
                        'is_split': True
                    }))

                # Start new chunk with overlap
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
                'page_start': section['page_start'],
                'chunk_part': len(chunks) + 1,
                'is_split': True
            }))

        return chunks

    def chunk_by_context_aware_sections(self) -> Tuple[List[str], List[Dict[str, str]]]:
        """
        Chunk the PDF using context-aware section splitting.

        Returns:
        --------
        Tuple[List[str], List[Dict]]
            - List of chunk texts
            - List of chunk metadata
        """
        print("\n🔄 Detecting sections in document...")
        sections = self._detect_sections()
        print(f"✓ Detected {len(sections)} sections\n")

        chunks = []
        metadata = []

        print("🔄 Creating context-aware chunks...")

        for section in sections:
            # Build context header
            context_header = f"[Section: {section['title']}]"

            # Count tokens
            content_tokens = self._count_tokens(section['content'])
            header_tokens = self._count_tokens(context_header)

            # If section fits in one chunk, keep it intact
            if header_tokens + content_tokens <= self.max_chunk_tokens:
                chunk_text = context_header + '\n\n' + section['content']
                chunk_text = self._clean_content(chunk_text)

                if len(chunk_text) >= 50:
                    chunks.append(chunk_text)
                    metadata.append({
                        'title': section['title'],
                        'page_start': section['page_start'],
                        'tokens': self._count_tokens(chunk_text),
                        'file': self.file_name,
                        'is_split': False
                    })
            else:
                # Section too large, split intelligently
                split_chunks = self._split_large_section(section, context_header)

                for chunk_text, meta in split_chunks:
                    if len(chunk_text) >= 50:
                        chunks.append(chunk_text)
                        metadata.append({
                            **meta,
                            'tokens': self._count_tokens(chunk_text),
                            'file': self.file_name
                        })

        print(f"✓ Created {len(chunks)} chunks\n")
        return chunks, metadata

    def save_chunks_to_file(self, output_path: str = None):
        """Save chunks to a text file for review."""
        if output_path is None:
            output_path = f"output_context_aware_pdf_chunks.txt"

        chunks, metadata = self.chunk_by_context_aware_sections()

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("CONTEXT-AWARE PDF CHUNKING RESULTS\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Source File: {self.pdf_file_path}\n")
            f.write(f"Max Chunk Tokens: {self.max_chunk_tokens}\n")
            f.write(f"Total Chunks: {len(chunks)}\n")
            f.write(f"OCR Used: {'Yes' if self.use_ocr else 'No'}\n")
            f.write("=" * 80 + "\n\n")

            for i, (chunk, meta) in enumerate(zip(chunks, metadata), 1):
                f.write(f"\n{'='*80}\n")
                f.write(f"CHUNK #{i}\n")
                f.write(f"{'='*80}\n")
                f.write(f"Section: {meta['title']}\n")
                f.write(f"Starting Page: {meta['page_start']}\n")
                f.write(f"Tokens: {meta['tokens']}\n")
                f.write(f"Split: {'Yes (Part ' + str(meta.get('chunk_part', 1)) + ')' if meta['is_split'] else 'No'}\n")
                f.write(f"{'-'*80}\n")
                f.write(chunk)
                f.write(f"\n{'-'*80}\n")

        print(f"\n✓ Saved {len(chunks)} chunks to: {output_path}")
        print(f"\n📊 Chunking Statistics:")
        print(f"  - Total Chunks: {len(chunks)}")
        print(f"  - Split Chunks: {sum(1 for m in metadata if m['is_split'])}")
        print(f"  - Intact Sections: {sum(1 for m in metadata if not m['is_split'])}")
        print(f"  - Average Tokens: {sum(m['tokens'] for m in metadata) // len(metadata)}")
        print(f"  - Min Tokens: {min(m['tokens'] for m in metadata)}")
        print(f"  - Max Tokens: {max(m['tokens'] for m in metadata)}")


def main():
    """Main function to demonstrate context-aware PDF chunking."""
    print("=" * 80)
    print("CONTEXT-AWARE PDF CHUNKER")
    print("=" * 80)
    print("\nIntelligent PDF chunker that:")
    print("  • Extracts text from PDFs (with OCR support)")
    print("  • Detects document structure and sections")
    print("  • Splits large sections while preserving context")
    print("  • Keeps small sections intact")
    print("  • Maintains semantic coherence\n")

    # Path to PDF file (in parent directory)
    pdf_file_path = r"C:\Personal\Mizan AI\Chunking\MizanAiChunking\Digital Government Policies - V2.0.pdf"

    print(f"📄 Processing: {pdf_file_path}\n")

    # Create chunker
    chunker = ContextAwarePDFChunker(
        pdf_file_path=pdf_file_path,
        max_chunk_tokens=512,
        overlap_tokens=50,
        use_ocr=True
    )

    # Get chunks
    chunks, metadata = chunker.chunk_by_context_aware_sections()

    # Preview first 3 chunks
    print("=" * 80)
    print("PREVIEW: First 3 Chunks")
    print("=" * 80)

    for i, (chunk, meta) in enumerate(zip(chunks[:3], metadata[:3]), 1):
        print(f"\n--- Chunk #{i} ---")
        print(f"Section: {meta['title']}")
        print(f"Page: {meta['page_start']}")
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
