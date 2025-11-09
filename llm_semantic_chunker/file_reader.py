"""
File Reader Module
==================

Extracts text, tables, and images from various file formats:
- PDF (.pdf)
- Word Documents (.docx)
- Text Files (.txt)
- Markdown Files (.md)

Installation:
pip install pypdf2 python-docx pillow pdf2image markdown beautifulsoup4 pdfplumber easyocr
"""

import os
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import base64
from io import BytesIO

# EasyOCR for image text extraction
try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False
    print("Warning: EasyOCR not installed. Install with: pip install easyocr")

# For PDF image extraction
try:
    from PIL import Image
    import io
    import numpy as np
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False


def read_txt_file(file_path: str) -> Dict:
    """
    Read text from .txt file.

    Parameters:
    -----------
    file_path : str
        Path to the .txt file

    Returns:
    --------
    Dict with keys: 'text', 'tables', 'images'
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()

        return {
            'text': text,
            'tables': [],
            'images': [],
            'metadata': {'file_type': 'txt', 'file_name': os.path.basename(file_path)}
        }
    except Exception as e:
        raise Exception(f"Error reading TXT file: {e}")


def read_md_file(file_path: str) -> Dict:
    """
    Read text from .md (Markdown) file.

    Parameters:
    -----------
    file_path : str
        Path to the .md file

    Returns:
    --------
    Dict with keys: 'text', 'tables', 'images'
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()

        # Try to parse markdown tables if markdown library is available
        tables = []
        try:
            import re
            # Find markdown tables
            table_pattern = r'\|(.+)\|[\r\n]+\|[-:\s|]+\|[\r\n]+((?:\|.+\|[\r\n]*)+)'
            table_matches = re.findall(table_pattern, text)

            for match in table_matches:
                tables.append(f"|{match[0]}|\n|---|\n{match[1]}")
        except:
            pass

        return {
            'text': text,
            'tables': tables,
            'images': [],
            'metadata': {'file_type': 'md', 'file_name': os.path.basename(file_path)}
        }
    except Exception as e:
        raise Exception(f"Error reading MD file: {e}")


def read_docx_file(file_path: str) -> Dict:
    """
    Read text, tables, and images from .docx file.

    Parameters:
    -----------
    file_path : str
        Path to the .docx file

    Returns:
    --------
    Dict with keys: 'text', 'tables', 'images'
    """
    try:
        from docx import Document
        from docx.oxml.table import CT_Tbl
        from docx.oxml.text.paragraph import CT_P
        from docx.table import _Cell, Table
        from docx.text.paragraph import Paragraph
    except ImportError:
        raise ImportError(
            "python-docx not installed. Install with: pip install python-docx"
        )

    try:
        doc = Document(file_path)

        # Extract text
        full_text = []
        tables = []
        images = []

        # Iterate through document elements maintaining order
        for element in doc.element.body:
            if isinstance(element, CT_P):
                # It's a paragraph
                para = Paragraph(element, doc)
                full_text.append(para.text)

            elif isinstance(element, CT_Tbl):
                # It's a table
                table = Table(element, doc)
                table_data = []
                for row in table.rows:
                    row_data = [cell.text.strip() for cell in row.cells]
                    table_data.append(row_data)

                # Format table as text
                table_text = "\n".join([" | ".join(row) for row in table_data])
                tables.append(table_text)
                full_text.append(f"\n[TABLE]\n{table_text}\n[/TABLE]\n")

        # Extract images
        try:
            for rel in doc.part.rels.values():
                if "image" in rel.target_ref:
                    image_data = rel.target_part.blob
                    images.append({
                        'data': image_data,
                        'format': rel.target_ref.split('.')[-1]
                    })
        except Exception as img_error:
            print(f"Warning: Could not extract images: {img_error}")

        return {
            'text': '\n'.join(full_text),
            'tables': tables,
            'images': images,
            'metadata': {
                'file_type': 'docx',
                'file_name': os.path.basename(file_path),
                'num_tables': len(tables),
                'num_images': len(images)
            }
        }
    except Exception as e:
        raise Exception(f"Error reading DOCX file: {e}")


def _fix_text_spacing(text: str) -> str:
    """
    Fix spacing issues in extracted text.
    Handles cases where words are concatenated without spaces.

    Common patterns:
    - wordWord -> word Word (lowercase followed by uppercase)
    - word.Word -> word. Word (punctuation followed by uppercase)
    - word123Word -> word123 Word (digits followed by uppercase)
    """
    import re

    # Pattern 1: Lowercase letter followed by uppercase letter
    # Example: "wordWord" -> "word Word"
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)

    # Pattern 2: Digit followed by uppercase letter
    # Example: "2024Digital" -> "2024 Digital"
    text = re.sub(r'(\d)([A-Z][a-z])', r'\1 \2', text)

    # Pattern 3: Lowercase followed by digit (less common but happens)
    # Example: "version2" -> "version 2" (only if digit is followed by space or uppercase)
    text = re.sub(r'([a-z])(\d+)([A-Z\s])', r'\1 \2\3', text)

    # Pattern 4: Punctuation without space before uppercase
    # Example: "end.Start" -> "end. Start"
    text = re.sub(r'([.!?])([A-Z])', r'\1 \2', text)

    return text


def read_pdf_file(file_path: str, enable_ocr: bool = True) -> Dict:
    """
    Read text, tables, and images from .pdf file with optional OCR.

    Parameters:
    -----------
    file_path : str
        Path to the .pdf file
    enable_ocr : bool, default=True
        Enable OCR to extract text from images (requires easyocr)

    Returns:
    --------
    Dict with keys: 'text', 'tables', 'images'
    """
    try:
        import pdfplumber
    except ImportError:
        raise ImportError(
            "pdfplumber not installed. Install with: pip install pdfplumber"
        )

    # Initialize EasyOCR reader if enabled and available
    ocr_reader = None
    if enable_ocr and EASYOCR_AVAILABLE:
        try:
            # Initialize with English and Arabic for KSA documents
            ocr_reader = easyocr.Reader(['en', 'ar'], gpu=False)
            print("EasyOCR initialized with English and Arabic support")
        except Exception as e:
            print(f"Warning: Could not initialize EasyOCR: {e}")
            ocr_reader = None
    elif enable_ocr and not EASYOCR_AVAILABLE:
        print("Warning: OCR requested but EasyOCR not installed. Install with: pip install easyocr")

    try:
        full_text = []
        tables = []
        images = []
        ocr_text_count = 0

        with pdfplumber.open(file_path) as pdf:
            for page_num, page in enumerate(pdf.pages):
                # Extract text with layout mode for better spacing
                # x_tolerance: horizontal tolerance for detecting separate words (default is 3)
                # y_tolerance: vertical tolerance for detecting separate lines (default is 3)
                # layout: preserve horizontal positioning (helps with spacing)
                page_text = page.extract_text(
                    x_tolerance=2,      # Smaller tolerance = better word separation
                    y_tolerance=3,
                    layout=False,       # False for continuous text with proper spacing
                    x_density=7.25,     # Characters per horizontal unit
                    y_density=13        # Characters per vertical unit
                )
                if page_text:
                    # Post-process to fix any remaining spacing issues
                    page_text = _fix_text_spacing(page_text)
                    full_text.append(f"\n--- Page {page_num + 1} ---\n")
                    full_text.append(page_text)

                # Extract tables
                page_tables = page.extract_tables()
                for table in page_tables:
                    if table:
                        # Filter out None values and format
                        table_cleaned = []
                        for row in table:
                            cleaned_row = [str(cell) if cell is not None else '' for cell in row]
                            table_cleaned.append(cleaned_row)

                        table_text = "\n".join([" | ".join(row) for row in table_cleaned])
                        tables.append(table_text)
                        full_text.append(f"\n[TABLE]\n{table_text}\n[/TABLE]\n")

                # First, check if this page has valid images (with proper dimensions)
                page_has_images = False
                valid_image_count = 0
                try:
                    if hasattr(page, 'images') and page.images:
                        # Extract image metadata and check if valid
                        for img in page.images:
                            # Check if image has valid dimensions (not just a tiny artifact)
                            x0 = img.get('x0', 0)
                            y0 = img.get('y0', 0)
                            x1 = img.get('x1', 0)
                            y1 = img.get('y1', 0)

                            # Calculate image size
                            width = abs(x1 - x0)
                            height = abs(y1 - y0)

                            # Only consider as real image if larger than 50x50 pixels
                            # This filters out small logos/headers (e.g., 72.7 x 34.3 header logo)
                            if width > 50 and height > 50:
                                page_has_images = True
                                valid_image_count += 1
                                images.append({
                                    'page': page_num + 1,
                                    'x0': x0,
                                    'y0': y0,
                                    'x1': x1,
                                    'y1': y1,
                                    'width': width,
                                    'height': height
                                })
                except Exception as img_error:
                    print(f"Warning: Could not extract image metadata from page {page_num + 1}: {img_error}")

                # Only run OCR on pages that have valid images
                if page_has_images and ocr_reader and PIL_AVAILABLE:
                    try:
                        print(f"   OCR processing page {page_num + 1} ({valid_image_count} image(s))...")

                        # Convert page to image for OCR (150 DPI for faster processing)
                        page_image = page.to_image(resolution=150)
                        pil_image = page_image.original

                        # Convert PIL Image to numpy array (required by EasyOCR)
                        image_np = np.array(pil_image)

                        # Perform OCR on the entire page image
                        ocr_results = ocr_reader.readtext(image_np)

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
                                full_text.append(f"\n[OCR_TEXT_PAGE_{page_num + 1}]\n{ocr_combined}\n[/OCR_TEXT]\n")
                                ocr_text_count += len(ocr_texts)
                                print(f"   ✓ Extracted {len(ocr_texts)} text segments from page {page_num + 1}")

                    except Exception as ocr_error:
                        print(f"Warning: OCR failed for page {page_num + 1}: {ocr_error}")

        if ocr_text_count > 0:
            print(f"Extracted {ocr_text_count} text segments from images using OCR")

        return {
            'text': '\n'.join(full_text),
            'tables': tables,
            'images': images,
            'metadata': {
                'file_type': 'pdf',
                'file_name': os.path.basename(file_path),
                'num_pages': len(pdf.pages),
                'num_tables': len(tables),
                'num_images': len(images),
                'ocr_segments': ocr_text_count
            }
        }
    except Exception as e:
        raise Exception(f"Error reading PDF file: {e}")


def read_file(file_path: str, enable_ocr: bool = True) -> Dict:
    """
    Universal file reader. Automatically detects file type and extracts content.

    Supported formats: .txt, .md, .docx, .pdf

    Parameters:
    -----------
    file_path : str
        Path to the file
    enable_ocr : bool, default=True
        Enable OCR for PDF files to extract text from images (requires easyocr)

    Returns:
    --------
    Dict with keys:
        - 'text': str - Full extracted text (includes OCR text if enabled)
        - 'tables': List[str] - List of tables as formatted strings
        - 'images': List - List of image data
        - 'metadata': Dict - File metadata

    Example:
    --------
    >>> content = read_file("document.pdf", enable_ocr=True)
    >>> print(content['text'])
    >>> print(f"Found {len(content['tables'])} tables")
    >>> print(f"Found {len(content['images'])} images")
    >>> print(f"OCR segments: {content['metadata'].get('ocr_segments', 0)}")
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    file_ext = Path(file_path).suffix.lower()

    if file_ext == '.txt':
        return read_txt_file(file_path)
    elif file_ext == '.md':
        return read_md_file(file_path)
    elif file_ext == '.docx':
        return read_docx_file(file_path)
    elif file_ext == '.pdf':
        return read_pdf_file(file_path, enable_ocr=enable_ocr)
    else:
        raise ValueError(
            f"Unsupported file format: {file_ext}. "
            f"Supported formats: .txt, .md, .docx, .pdf"
        )


def get_full_text(file_path: str, enable_ocr: bool = True) -> str:
    """
    Quick helper to get just the text content from a file.

    Parameters:
    -----------
    file_path : str
        Path to the file
    enable_ocr : bool, default=True
        Enable OCR for PDF files to extract text from images

    Returns:
    --------
    str
        Extracted text content (includes OCR text if enabled)

    Example:
    --------
    >>> text = get_full_text("document.pdf", enable_ocr=True)
    >>> print(text)
    """
    content = read_file(file_path, enable_ocr=enable_ocr)
    return content['text']


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("FILE READER TEST")
    print("=" * 80)

    # Test with a sample file (you'll need to provide a real file path)
    test_file = "sample.txt"

    if os.path.exists(test_file):
        try:
            content = read_file(test_file)

            print(f"\n📄 File: {content['metadata']['file_name']}")
            print(f"📝 Type: {content['metadata']['file_type']}")
            print(f"📊 Tables found: {len(content['tables'])}")
            print(f"🖼️  Images found: {len(content['images'])}")
            print(f"\n📖 Text preview (first 500 chars):")
            print(content['text'][:500])

            if content['tables']:
                print(f"\n📊 First table:")
                print(content['tables'][0][:200])

        except Exception as e:
            print(f"Error: {e}")
    else:
        print(f"\n⚠️  Sample file '{test_file}' not found.")
        print("Create a test file or modify the test_file variable.")
