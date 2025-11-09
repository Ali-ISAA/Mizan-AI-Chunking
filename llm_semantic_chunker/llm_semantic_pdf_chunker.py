"""
LLM Semantic PDF Chunker
=========================

Uses Google Gemini LLM to intelligently chunk PDF files based PURELY on semantic meaning.
The LLM decides ALL chunk boundaries - no size limits!

Features:
- Extracts text from PDFs with OCR support (English + Arabic)
- Uses Gemini LLM to determine ALL split points based on meaning
- LLM analyzes thematic shifts and decides boundaries
- No artificial size constraints - chunks can be any size
- Detects document structure (sections)

Strategy:
1. Extract text from PDF (with OCR for images)
2. Detect sections in the document
3. For each section, let LLM decide ALL split points
4. LLM groups semantically related sentences together
5. Chunks are created based purely on meaning, not size
"""

import re
import sys
import tiktoken
import pdfplumber
import numpy as np
import google.generativeai as genai
from typing import List, Dict, Tuple
from pathlib import Path

# Add parent directory to path to import from other modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from other_chunkers.text_chunking_methods import GEMINI_API_KEYS, get_gemini_embeddings
from llm_semantic_chunker.api_key_manager import get_key_manager

# OCR support
try:
    import easyocr
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False


class LLMSemanticPDFChunker:
    """
    Intelligent PDF chunker using LLM for pure semantic analysis.
    """

    def __init__(
        self,
        pdf_file_path: str,
        model_name: str = "gemini-2.0-flash-lite",
        temperature: float = 0.2,
        use_ocr: bool = True
    ):
        """
        Initialize the LLM semantic PDF chunker.

        Parameters:
        -----------
        pdf_file_path : str
            Path to the PDF file
        model_name : str
            Gemini model to use (default: gemini-2.0-flash-lite)
        temperature : float
            LLM temperature for consistency (default: 0.2)
        use_ocr : bool
            Whether to use OCR for images (default: True)
        """
        self.pdf_file_path = pdf_file_path
        self.file_name = Path(pdf_file_path).stem
        self.encoding = tiktoken.get_encoding("cl100k_base")
        self.use_ocr = use_ocr and OCR_AVAILABLE

        # Initialize OCR if available
        self.ocr_reader = None
        if self.use_ocr:
            print("🔄 Initializing EasyOCR (English + Arabic)...")
            self.ocr_reader = easyocr.Reader(['en', 'ar'], gpu=False)
            print("✓ EasyOCR initialized")

        # Extract text from PDF
        self.content = self._extract_text_from_pdf()

        # Initialize Gemini model
        self.model = genai.GenerativeModel(
            model_name,
            generation_config=genai.types.GenerationConfig(
                temperature=temperature,
                max_output_tokens=500,
            )
        )

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
            if width > 50 and height > 50:
                return True
        return False

    def _extract_text_from_image(self, page, page_num) -> str:
        """Extract text from images in page using OCR."""
        if not self.ocr_reader:
            return ""

        try:
            page_image = page.to_image(resolution=150)
            pil_image = page_image.original
            image_np = np.array(pil_image)

            ocr_results = self.ocr_reader.readtext(image_np)

            if ocr_results:
                ocr_texts = []
                for detection in ocr_results:
                    text = detection[1]
                    confidence = detection[2]
                    if confidence > 0.3:
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
        text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
        text = re.sub(r'(\d)([A-Z][a-z])', r'\1 \2', text)
        text = re.sub(r'([a-z])(\d+)([A-Z\s])', r'\1 \2\3', text)
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

                # Check for images and run OCR
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
        """Clean content by removing page markers and extra whitespace."""
        text = re.sub(r'--- Page \d+ ---\s*', '', text)
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = text.strip()
        return text

    def _detect_sections(self) -> List[Dict[str, str]]:
        """
        Detect sections in PDF text using patterns.

        Returns:
        --------
        List[Dict]
            List of sections with metadata
        """
        sections = []
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

                # Pattern 1: Numbered sections (e.g., "01. Title", "1. Title")
                section_match = re.match(r'^(\d{1,2})\s*[.\)]\s*(.+)$', line_stripped)

                # Pattern 2: ALL CAPS headings
                caps_match = re.match(r'^([A-Z][A-Z\s]{10,})$', line_stripped)

                if section_match and len(section_match.group(2)) > 5:
                    if current_section['content']:
                        section_text = '\n'.join(current_section['content']).strip()
                        if section_text and len(section_text) > 50:
                            current_section['content'] = section_text
                            sections.append(current_section.copy())

                    title = section_match.group(2).strip()
                    current_section = {
                        'title': f"{section_match.group(1)}. {title}",
                        'content': [],
                        'page_start': current_page
                    }

                elif caps_match and len(caps_match.group(1).split()) >= 2:
                    if current_section['content']:
                        section_text = '\n'.join(current_section['content']).strip()
                        if section_text and len(section_text) > 50:
                            current_section['content'] = section_text
                            sections.append(current_section.copy())

                    current_section = {
                        'title': caps_match.group(1).strip(),
                        'content': [],
                        'page_start': current_page
                    }

                else:
                    if line_stripped:
                        current_section['content'].append(line)

            current_page += 1

        # Last section
        if current_section['content']:
            section_text = '\n'.join(current_section['content']).strip()
            if section_text and len(section_text) > 50:
                current_section['content'] = section_text
                sections.append(current_section.copy())

        return sections

    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]

    def _llm_find_split_points(self, sentences: List[str], section_title: str) -> List[int]:
        """
        Use LLM to find optimal split points in sentences.

        Parameters:
        -----------
        sentences : List[str]
            List of sentences
        section_title : str
            Title of the section for context

        Returns:
        --------
        List[int]
            Indices where splits should occur
        """
        if len(sentences) <= 3:
            return []

        # Prepare input with sentence markers
        marked_text = ""
        for i, sent in enumerate(sentences):
            marked_text += f"<|sent_{i}|>{sent}<|/sent_{i}|> "

        # Create prompt for PDF document analysis
        prompt = f"""You are analyzing a section titled "{section_title}" from a PDF document.

The text has been split into sentences, each marked with <|sent_X|> and <|/sent_X|> tags.

Your task: Identify where COMPLETE TOPIC BOUNDARIES occur. Each chunk should contain:
- A main heading/topic with ALL its related content
- Complete explanations, lists, and details about that topic
- NO splitting within a single concept or explanation

Guidelines:
- Keep related sentences that explain the SAME topic together
- Split ONLY when the content moves to a NEW main topic or concept
- Prefer LARGER chunks that keep complete information together
- A chunk should be self-contained and understandable on its own

For example:
- If sentences 0-5 explain "Policy A" and sentences 6-10 explain "Policy B", respond: split_after: 5
- If all sentences explain the same topic, respond: split_after: none

TEXT:
{marked_text}

Respond ONLY with format: split_after: 5, 12
If no clear topic boundaries exist (all content relates to same topic), respond: split_after: none"""

        try:
            manager = get_key_manager()

            def llm_call():
                response = self.model.generate_content(prompt)
                return response.text

            result_text = manager.execute_with_retry(llm_call)

            if 'none' in result_text.lower():
                return []

            split_line = [line for line in result_text.split('\n')
                         if 'split_after:' in line.lower()]

            if split_line:
                numbers_str = split_line[0].split('split_after:')[1]
                numbers = [int(n.strip()) for n in numbers_str.split(',') if n.strip().isdigit()]
                return [n for n in numbers if 0 <= n < len(sentences) - 1]

            return []

        except Exception as e:
            print(f"⚠️  LLM split analysis failed: {e}")
            return []

    def _chunk_section_with_llm(self, section: Dict) -> List[Tuple[str, Dict]]:
        """
        Chunk a section using PURE LLM semantic analysis - no size limits!
        """
        content = section['content']

        # Build context header
        context_header = f"[Section: {section['title']}]"

        # Always use LLM to analyze
        print(f"  🤖 Using LLM to analyze: {section['title']}")

        sentences = self._split_into_sentences(content)

        if len(sentences) <= 2:
            chunk_text = context_header + '\n\n' + content
            chunk_text = self._clean_content(chunk_text)
            return [(chunk_text, {
                'title': section['title'],
                'page_start': section['page_start'],
                'split_by_llm': False
            })]

        # Get split points from LLM
        split_indices = self._llm_find_split_points(sentences, section['title'])

        if not split_indices:
            # LLM says no splits needed
            chunk_text = context_header + '\n\n' + content
            chunk_text = self._clean_content(chunk_text)
            return [(chunk_text, {
                'title': section['title'],
                'page_start': section['page_start'],
                'split_by_llm': True,
                'llm_decision': 'no_split_needed'
            })]

        # Create chunks based on LLM's split points
        chunks = []
        split_indices = sorted(set([0] + split_indices + [len(sentences)]))

        for i in range(len(split_indices) - 1):
            start_idx = split_indices[i]
            end_idx = split_indices[i + 1]

            chunk_sentences = sentences[start_idx:end_idx]
            chunk_content = ' '.join(chunk_sentences)

            chunk_text = context_header + '\n\n' + chunk_content
            chunk_text = self._clean_content(chunk_text)

            if len(chunk_text) >= 50:
                chunks.append((chunk_text, {
                    'title': section['title'],
                    'page_start': section['page_start'],
                    'split_by_llm': True,
                    'chunk_part': i + 1
                }))

        return chunks if chunks else [(context_header + '\n\n' + content, {
            'title': section['title'],
            'page_start': section['page_start'],
            'split_by_llm': True,
            'llm_decision': 'fallback'
        })]

    def chunk_with_llm_semantics(self) -> Tuple[List[str], List[Dict]]:
        """
        Chunk PDF using PURE LLM semantic analysis - no size limits!

        Returns:
        --------
        Tuple[List[str], List[Dict]]
            - List of chunk texts
            - List of chunk metadata
        """
        print(f"\n📄 Processing: {self.pdf_file_path}")
        print(f"🎯 LLM decides ALL boundaries - no size limits!\n")

        sections = self._detect_sections()
        print(f"✓ Detected {len(sections)} sections\n")

        chunks = []
        metadata = []

        print("🤖 Analyzing sections with Gemini LLM...\n")

        for section in sections:
            section_chunks = self._chunk_section_with_llm(section)

            for chunk_text, meta in section_chunks:
                if len(chunk_text) >= 50:
                    chunks.append(chunk_text)
                    metadata.append({
                        **meta,
                        'tokens': self._count_tokens(chunk_text),
                        'file': self.file_name
                    })

        print(f"\n✓ Created {len(chunks)} semantic chunks")
        return chunks, metadata

    def save_chunks_to_file(self, output_path: str = None):
        """Save chunks to a text file for review."""
        if output_path is None:
            output_path = f"output_llm_semantic_pdf_chunks.txt"

        chunks, metadata = self.chunk_with_llm_semantics()

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("LLM SEMANTIC PDF CHUNKING RESULTS\n")
            f.write("(LLM Decides ALL Boundaries - No Size Limits!)\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Source File: {self.pdf_file_path}\n")
            f.write(f"Chunking Method: Pure LLM semantic analysis\n")
            f.write(f"OCR Used: {'Yes' if self.use_ocr else 'No'}\n")
            f.write(f"Total Chunks: {len(chunks)}\n")
            f.write("=" * 80 + "\n\n")

            for i, (chunk, meta) in enumerate(zip(chunks, metadata), 1):
                f.write(f"\n{'='*80}\n")
                f.write(f"CHUNK #{i}\n")
                f.write(f"{'='*80}\n")
                f.write(f"Section: {meta['title']}\n")
                f.write(f"Starting Page: {meta['page_start']}\n")
                f.write(f"Tokens: {meta['tokens']}\n")
                f.write(f"Split by LLM: {'Yes' if meta['split_by_llm'] else 'No'}\n")
                if meta.get('chunk_part'):
                    f.write(f"Part: {meta['chunk_part']}\n")
                f.write(f"{'-'*80}\n")
                f.write(chunk)
                f.write(f"\n{'-'*80}\n")

        print(f"\n✓ Saved {len(chunks)} chunks to: {output_path}")
        print(f"\n📊 Chunking Statistics:")
        print(f"  - Total Chunks: {len(chunks)}")
        print(f"  - LLM-Split Chunks: {sum(1 for m in metadata if m['split_by_llm'])}")
        print(f"  - Average Tokens: {sum(m['tokens'] for m in metadata) // len(metadata)}")
        print(f"  - Min Tokens: {min(m['tokens'] for m in metadata)}")
        print(f"  - Max Tokens: {max(m['tokens'] for m in metadata)}")


def main():
    """Main function to demonstrate LLM semantic PDF chunking."""
    print("=" * 80)
    print("LLM SEMANTIC PDF CHUNKER")
    print("(Pure LLM-Driven - No Size Limits!)")
    print("=" * 80)
    print("\nIntelligent PDF chunker that:")
    print("  • Extracts text from PDFs with OCR support")
    print("  • LLM decides ALL chunk boundaries based on meaning")
    print("  • NO artificial size constraints")
    print("  • Splits ONLY at thematic shifts")
    print("  • Chunks can be any size - small or large!\n")

    # Path to PDF file
    pdf_file_path = r"C:\Personal\Mizan AI\Chunking\MizanAiChunking\Digital Government Policies - V2.0.pdf"

    print(f"📄 Processing: {pdf_file_path}\n")

    # Create chunker
    chunker = LLMSemanticPDFChunker(
        pdf_file_path=pdf_file_path,
        model_name="gemini-2.0-flash-lite",
        temperature=0.2,
        use_ocr=True
    )

    # Get chunks
    chunks, metadata = chunker.chunk_with_llm_semantics()

    # Preview first 3 chunks
    print("\n" + "=" * 80)
    print("PREVIEW: First 3 Chunks")
    print("=" * 80)

    for i, (chunk, meta) in enumerate(zip(chunks[:3], metadata[:3]), 1):
        print(f"\n--- Chunk #{i} ---")
        print(f"Section: {meta['title']}")
        print(f"Page: {meta['page_start']}")
        print(f"Tokens: {meta['tokens']}")
        print(f"Split by LLM: {'Yes' if meta['split_by_llm'] else 'No'}")
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
