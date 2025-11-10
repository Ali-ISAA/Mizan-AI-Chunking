"""
File reader utility for multiple document types
"""

import re
from pathlib import Path
from typing import Dict, Optional


def read_file(file_path: str) -> Dict:
    """
    Read file and extract text content

    Supports: .txt, .md, .pdf, .docx

    Parameters:
    -----------
    file_path : str
        Path to the file

    Returns:
    --------
    Dict
        Dictionary with 'text', 'file_name', 'file_type'
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    file_type = path.suffix.lower()
    file_name = path.stem

    if file_type in ['.txt', '.md']:
        text = _read_text_file(file_path)
    elif file_type == '.pdf':
        text = _read_pdf_file(file_path)
    elif file_type == '.docx':
        text = _read_docx_file(file_path)
    else:
        raise ValueError(f"Unsupported file type: {file_type}")

    return {
        'text': text,
        'file_name': file_name,
        'file_type': file_type
    }


def get_file_text(file_path: str) -> str:
    """
    Get just the text content from a file

    Parameters:
    -----------
    file_path : str
        Path to the file

    Returns:
    --------
    str
        Text content
    """
    return read_file(file_path)['text']


def _read_text_file(file_path: str) -> str:
    """Read text or markdown file"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()


def _read_pdf_file(file_path: str) -> str:
    """Read PDF file"""
    try:
        import pdfplumber
    except ImportError:
        raise ImportError("pdfplumber is required for PDF files. Install: pip install pdfplumber")

    text = []
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text.append(page_text)

    return '\n\n'.join(text)


def _read_docx_file(file_path: str) -> str:
    """Read DOCX file"""
    try:
        from docx import Document
    except ImportError:
        raise ImportError("python-docx is required for DOCX files. Install: pip install python-docx")

    doc = Document(file_path)
    text = []

    for para in doc.paragraphs:
        if para.text.strip():
            text.append(para.text)

    return '\n\n'.join(text)


def clean_text(text: str) -> str:
    """
    Clean text by removing extra whitespace and special characters

    Parameters:
    -----------
    text : str
        Raw text

    Returns:
    --------
    str
        Cleaned text
    """
    # Remove image placeholders
    text = re.sub(r'<!-- image -->\s*', '', text)

    # Remove multiple consecutive newlines
    text = re.sub(r'\n{3,}', '\n\n', text)

    # Remove leading/trailing whitespace
    text = text.strip()

    return text
