"""
Text Chunking Methods with Google Gemini Embeddings
====================================================

This module provides 5 text chunking strategies for documents (PDF, DOCX, TXT, MD),
all using Google Gemini API for embeddings and LLM operations.

Google API Configuration:
-------------------------
API Key: AIzaSyABM0dK6NCjVN7siU2dwxwY9aPnFhIwbTU
Name: EMBEDDINGS
Project: gen-lang-client-0837634806
Project Number: 280248277567

Installation Requirements:
--------------------------
pip install google-generativeai tiktoken numpy pdfplumber python-docx pillow

Available Methods:
------------------
1. fixed_token_chunking() - Fixed-size token chunks
2. recursive_token_chunking() - Recursive splitting by separators
3. cluster_semantic_chunking() - Clustering-based semantic chunks
4. kamradt_semantic_chunking() - Similarity-based semantic chunks
5. llm_semantic_chunking() - Gemini LLM-based intelligent chunks

All methods accept file_path and return: (chunks, embeddings)
    - chunks: List[str] - The text chunks
    - embeddings: List[List[float]] - Gemini embeddings for each chunk
"""

import os
import re
import sys
import time
from typing import List, Optional, Callable, Tuple, Dict
from pathlib import Path
import numpy as np
import tiktoken

# Import file reader from llm_semantic_chunker
sys.path.insert(0, str(Path(__file__).parent.parent / "llm_semantic_chunker"))
from file_reader import read_file, get_full_text

try:
    import google.generativeai as genai
except ImportError:
    raise ImportError(
        "Google Generative AI library not found. "
        "Install it with: pip install google-generativeai"
    )

# ============================================================================
# GOOGLE API CONFIGURATION - Multiple Keys for Rate Limit Rotation
# ============================================================================

# Load API keys from environment variables
from dotenv import load_dotenv

# Load .env from project root
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

# Multiple API keys for automatic rotation when rate limits are hit
GEMINI_API_KEYS = [
    os.getenv('GEMINI_API_KEY_1'),
    os.getenv('GEMINI_API_KEY_2'),
    os.getenv('GEMINI_API_KEY_3'),
    os.getenv('GEMINI_API_KEY_4'),
]

# Filter out None values (in case not all keys are set)
GEMINI_API_KEYS = [key for key in GEMINI_API_KEYS if key]

if not GEMINI_API_KEYS:
    raise ValueError("No API keys found! Please set GEMINI_API_KEY_1, GEMINI_API_KEY_2, etc. in .env file")

PROJECT_NAME = "projects/280248277567"
PROJECT_NUMBER = 280248277567
PROJECT_ID = "gen-lang-client-0837634806"

# Note: API key manager is initialized in llm_semantic_md_chunker.py
# We import it here for use in the chunking functions
sys.path.insert(0, str(Path(__file__).parent.parent / "llm_semantic_chunker"))
from api_key_manager import get_key_manager


# ============================================================================
# Utility Functions
# ============================================================================

def get_gemini_embeddings(
    texts: List[str],
    model_name: str = "models/embedding-001",
    batch_size: int = 100
) -> List[List[float]]:
    """
    Generate embeddings for texts using Google Gemini with automatic key rotation.

    FREE TIER LIMITS (per key):
    - RPM: 100 requests/minute
    - TPM: 30,000 tokens/minute
    - RPD: 1,000 requests/day

    Parameters:
    -----------
    texts : List[str]
        List of texts to embed
    model_name : str
        Gemini embedding model name
    batch_size : int
        Number of texts to process in one batch

    Returns:
    --------
    List[List[float]]
        List of embedding vectors
    """
    all_embeddings = []
    # Get or initialize key manager with API keys
    manager = get_key_manager(api_keys=GEMINI_API_KEYS)

    # Process in batches to avoid rate limits
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]

        batch_embeddings = []
        for idx, text in enumerate(batch):
            def embed_single_text():
                # Truncate very long texts
                truncated_text = text[:10000] if len(text) > 10000 else text

                result = genai.embed_content(
                    model=model_name,
                    content=truncated_text,
                    task_type="retrieval_document"
                )
                return result['embedding']

            try:
                # Use key manager to handle rate limits automatically
                embedding = manager.execute_with_retry(embed_single_text)
                batch_embeddings.append(embedding)

                # Delay to respect 100 RPM limit (90 requests/min = 0.67s delay)
                time.sleep(0.67)

            except Exception as e:
                error_msg = str(e)
                # Check if all keys are exhausted
                if "429" in error_msg or "quota" in error_msg.lower():
                    print(f"\n⚠️  All {len(GEMINI_API_KEYS)} API keys exhausted!")
                    print(f"   Processed {i + idx}/{len(texts)} texts before hitting limits")
                    print(f"   Quota resets at midnight PST")
                    print(f"   Check usage: https://ai.dev/usage")
                    # Return zero vector for remaining texts
                    batch_embeddings.append([0.0] * 768)
                else:
                    print(f"Warning: Failed to embed text (length {len(text)}): {e}")
                    batch_embeddings.append([0.0] * 768)

        all_embeddings.extend(batch_embeddings)

    return all_embeddings


def openai_token_count(text: str) -> int:
    """Count tokens using tiktoken (OpenAI tokenizer)."""
    try:
        encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))
    except:
        # Fallback to approximate count
        return len(text) // 4


# ============================================================================
# Internal Helper Functions
# ============================================================================

def _recursive_split_text(
    text: str,
    chunk_size: int,
    overlap: int,
    separators: List[str],
    keep_separator: bool = True
) -> List[str]:
    """Internal helper for recursive text splitting without file I/O or embeddings."""
    def split_text_with_regex(text: str, separator: str, keep_sep: bool) -> List[str]:
        if separator:
            if keep_sep:
                _splits = re.split(f"({re.escape(separator)})", text)
                splits = [_splits[i] + _splits[i + 1] for i in range(1, len(_splits), 2)]
                if len(_splits) % 2 == 0:
                    splits += _splits[-1:]
                splits = [_splits[0]] + splits
            else:
                splits = re.split(re.escape(separator), text)
        else:
            splits = list(text)
        return [s for s in splits if s != ""]

    def merge_splits(splits: List[str], separator: str) -> List[str]:
        docs = []
        current_doc = []
        total = 0
        separator_len = openai_token_count(separator)

        for split in splits:
            split_len = openai_token_count(split)

            if total + split_len + (separator_len if current_doc else 0) > chunk_size:
                if current_doc:
                    doc = separator.join(current_doc).strip()
                    if doc:
                        docs.append(doc)

                    while total > overlap or (
                        total + split_len + (separator_len if current_doc else 0) > chunk_size
                        and total > 0
                    ):
                        total -= openai_token_count(current_doc[0]) + (
                            separator_len if len(current_doc) > 1 else 0
                        )
                        current_doc = current_doc[1:]

            current_doc.append(split)
            total += split_len + (separator_len if len(current_doc) > 1 else 0)

        if current_doc:
            doc = separator.join(current_doc).strip()
            if doc:
                docs.append(doc)

        return docs

    def recursive_split(text: str, seps: List[str]) -> List[str]:
        final_chunks = []
        separator = seps[-1]
        new_separators = []

        for i, sep in enumerate(seps):
            if sep == "":
                separator = sep
                break
            if re.search(re.escape(sep), text):
                separator = sep
                new_separators = seps[i + 1:]
                break

        splits = split_text_with_regex(text, separator, keep_separator)

        good_splits = []
        sep_to_use = "" if keep_separator else separator

        for s in splits:
            if openai_token_count(s) < chunk_size:
                good_splits.append(s)
            else:
                if good_splits:
                    merged = merge_splits(good_splits, sep_to_use)
                    final_chunks.extend(merged)
                    good_splits = []

                if not new_separators:
                    final_chunks.append(s)
                else:
                    other_chunks = recursive_split(s, new_separators)
                    final_chunks.extend(other_chunks)

        if good_splits:
            merged = merge_splits(good_splits, sep_to_use)
            final_chunks.extend(merged)

        return final_chunks

    return recursive_split(text, separators)


# ============================================================================
# 1. FIXED TOKEN CHUNKING
# ============================================================================

def fixed_token_chunking(
    file_path: str,
    chunk_size: int = 512,
    overlap: int = 0,
    encoding_name: str = "cl100k_base",
    generate_embeddings: bool = False
) -> Tuple[List[str], List[List[float]]]:
    """
    Split document into fixed-size token chunks and generate embeddings.

    This method reads a file (PDF, DOCX, TXT, MD), extracts text, tokenizes it,
    and splits into equal-sized chunks with optional overlap.

    Parameters:
    -----------
    file_path : str
        Path to the document file (.pdf, .docx, .txt, .md)
    chunk_size : int, default=512
        Number of tokens per chunk
    overlap : int, default=0
        Number of overlapping tokens between chunks
    encoding_name : str, default="cl100k_base"
        Tiktoken encoding name

    Returns:
    --------
    Tuple[List[str], List[List[float]]]
        (text_chunks, embeddings)

    Example:
    --------
    >>> chunks, embeddings = fixed_token_chunking(
    ...     "document.pdf",
    ...     chunk_size=512,
    ...     overlap=50
    ... )
    >>> print(f"Created {len(chunks)} chunks")
    >>> print(f"Embedding dimension: {len(embeddings[0])}")
    """
    # Read file
    print(f"Reading file: {file_path}")
    content = read_file(file_path)
    text = content['text']

    print(f"Extracted {len(text)} characters")
    if content['tables']:
        print(f"Found {len(content['tables'])} tables")
    if content['images']:
        print(f"Found {len(content['images'])} images")

    # Get tokenizer
    encoding = tiktoken.get_encoding(encoding_name)

    # Count total tokens
    total_tokens = len(encoding.encode(text))
    print(f"Total tokens: {total_tokens}")

    # Create chunks with overlap - extract directly from original text
    # This preserves proper spacing and word boundaries
    chunks = []
    start_pos = 0

    while start_pos < len(text):
        # Estimate character length needed for target tokens (rough: 4 chars per token)
        estimated_end = start_pos + (chunk_size * 4)

        # Get candidate chunk from original text
        candidate = text[start_pos:min(estimated_end, len(text))]
        candidate_tokens = len(encoding.encode(candidate))

        # Adjust to hit target token count
        if candidate_tokens < chunk_size and estimated_end < len(text):
            # Need more characters
            while candidate_tokens < chunk_size and len(candidate) < len(text) - start_pos:
                estimated_end += 100
                candidate = text[start_pos:min(estimated_end, len(text))]
                candidate_tokens = len(encoding.encode(candidate))
        elif candidate_tokens > chunk_size:
            # Need fewer characters
            while candidate_tokens > chunk_size and len(candidate) > 100:
                estimated_end -= 100
                candidate = text[start_pos:estimated_end]
                candidate_tokens = len(encoding.encode(candidate))

        # Snap to word boundary (don't split mid-word)
        if estimated_end < len(text):
            # Find last space to avoid cutting words
            last_space = candidate.rfind(' ')
            last_newline = candidate.rfind('\n')
            snap_pos = max(last_space, last_newline)
            if snap_pos > len(candidate) // 2:  # Only snap if we're past halfway
                candidate = candidate[:snap_pos + 1]
                estimated_end = start_pos + snap_pos + 1

        chunks.append(candidate)

        # Calculate next start position with overlap
        if overlap > 0 and estimated_end < len(text):
            # Find character length of overlap tokens
            # Work backwards from end of current chunk
            overlap_start = start_pos
            overlap_chars = min(overlap * 4, len(candidate))  # Estimate
            overlap_start = max(start_pos, estimated_end - overlap_chars)

            # Fine-tune to get exactly 'overlap' tokens
            overlap_text = text[overlap_start:estimated_end]
            overlap_tokens = len(encoding.encode(overlap_text))

            # Adjust to get closer to target overlap
            while overlap_tokens < overlap and overlap_start > start_pos:
                overlap_start -= 20
                overlap_text = text[overlap_start:estimated_end]
                overlap_tokens = len(encoding.encode(overlap_text))

            while overlap_tokens > overlap and overlap_start < estimated_end - 50:
                overlap_start += 10
                overlap_text = text[overlap_start:estimated_end]
                overlap_tokens = len(encoding.encode(overlap_text))

            # Snap overlap start to word boundary
            if overlap_start > start_pos:
                substr = text[overlap_start:min(overlap_start + 50, estimated_end)]
                first_space = substr.find(' ')
                if first_space > 0 and first_space < 20:
                    overlap_start += first_space + 1

            start_pos = overlap_start
        else:
            start_pos = estimated_end

        # Safety check to avoid infinite loop
        if len(candidate) == 0 or start_pos >= len(text):
            break

    # Generate embeddings (optional)
    if generate_embeddings:
        print(f"Generating embeddings for {len(chunks)} chunks...")
        embeddings = get_gemini_embeddings(chunks)
        print(f"Complete! Created {len(chunks)} chunks with embeddings")
    else:
        embeddings = []
        print(f"Complete! Created {len(chunks)} chunks")

    return chunks, embeddings


# ============================================================================
# 2. RECURSIVE TOKEN CHUNKING
# ============================================================================

def recursive_token_chunking(
    file_path: str,
    chunk_size: int = 512,
    overlap: int = 0,
    separators: Optional[List[str]] = None,
    keep_separator: bool = True,
    generate_embeddings: bool = False
) -> Tuple[List[str], List[List[float]]]:
    """
    Split document recursively using hierarchy of separators.

    Reads file and tries to split at natural boundaries (paragraphs, sentences, words)
    while respecting chunk size limits.

    Parameters:
    -----------
    file_path : str
        Path to the document file (.pdf, .docx, .txt, .md)
    chunk_size : int, default=512
        Maximum tokens per chunk
    overlap : int, default=0
        Token overlap between chunks
    separators : List[str], optional
        Separators to try in order. Default: ["\\n\\n", "\\n", ".", "?", "!", " ", ""]
    keep_separator : bool, default=True
        Whether to keep separators in chunks

    Returns:
    --------
    Tuple[List[str], List[List[float]]]
        (text_chunks, embeddings)

    Example:
    --------
    >>> chunks, embeddings = recursive_token_chunking(
    ...     "document.docx",
    ...     chunk_size=512,
    ...     overlap=50
    ... )
    """
    # Read file
    print(f"Reading file: {file_path}")
    content = read_file(file_path)
    text = content['text']
    print(f"Extracted {len(text)} characters")

    if separators is None:
        separators = ["\n\n", "\n", ".", "?", "!", " ", ""]

    def split_text_with_regex(text: str, separator: str, keep_sep: bool) -> List[str]:
        if separator:
            if keep_sep:
                _splits = re.split(f"({re.escape(separator)})", text)
                splits = [_splits[i] + _splits[i + 1] for i in range(1, len(_splits), 2)]
                if len(_splits) % 2 == 0:
                    splits += _splits[-1:]
                splits = [_splits[0]] + splits
            else:
                splits = re.split(re.escape(separator), text)
        else:
            splits = list(text)
        return [s for s in splits if s != ""]

    def merge_splits(splits: List[str], separator: str) -> List[str]:
        docs = []
        current_doc = []
        total = 0
        separator_len = openai_token_count(separator)

        for split in splits:
            split_len = openai_token_count(split)

            if total + split_len + (separator_len if current_doc else 0) > chunk_size:
                if current_doc:
                    doc = separator.join(current_doc).strip()
                    if doc:
                        docs.append(doc)

                    # Handle overlap
                    while total > overlap or (
                        total + split_len + (separator_len if current_doc else 0) > chunk_size
                        and total > 0
                    ):
                        total -= openai_token_count(current_doc[0]) + (
                            separator_len if len(current_doc) > 1 else 0
                        )
                        current_doc = current_doc[1:]

            current_doc.append(split)
            total += split_len + (separator_len if len(current_doc) > 1 else 0)

        if current_doc:
            doc = separator.join(current_doc).strip()
            if doc:
                docs.append(doc)

        return docs

    def recursive_split(text: str, seps: List[str]) -> List[str]:
        final_chunks = []
        separator = seps[-1]
        new_separators = []

        for i, sep in enumerate(seps):
            if sep == "":
                separator = sep
                break
            if re.search(re.escape(sep), text):
                separator = sep
                new_separators = seps[i + 1:]
                break

        splits = split_text_with_regex(text, separator, keep_separator)

        good_splits = []
        sep_to_use = "" if keep_separator else separator

        for s in splits:
            if openai_token_count(s) < chunk_size:
                good_splits.append(s)
            else:
                if good_splits:
                    merged = merge_splits(good_splits, sep_to_use)
                    final_chunks.extend(merged)
                    good_splits = []

                if not new_separators:
                    final_chunks.append(s)
                else:
                    other_chunks = recursive_split(s, new_separators)
                    final_chunks.extend(other_chunks)

        if good_splits:
            merged = merge_splits(good_splits, sep_to_use)
            final_chunks.extend(merged)

        return final_chunks

    # Perform recursive splitting
    chunks = recursive_split(text, separators)

    # Generate embeddings (optional)
    if generate_embeddings:
        print(f"Generating embeddings for {len(chunks)} chunks...")
        embeddings = get_gemini_embeddings(chunks)
        print(f"Complete! Created {len(chunks)} chunks with embeddings")
    else:
        embeddings = []
        print(f"Complete! Created {len(chunks)} chunks")

    return chunks, embeddings


# ============================================================================
# 3. CLUSTER SEMANTIC CHUNKING
# ============================================================================

def cluster_semantic_chunking(
    file_path: str,
    max_chunk_size: int = 400,
    min_chunk_size: int = 50,
    overlap: int = 0,
    generate_embeddings: bool = False
) -> Tuple[List[str], List[List[float]]]:
    """
    Split document into semantically coherent chunks using clustering.

    Uses embeddings and dynamic programming to find optimal groupings
    based on semantic similarity.

    Parameters:
    -----------
    file_path : str
        Path to the document file (.pdf, .docx, .txt, .md)
    max_chunk_size : int, default=400
        Maximum tokens per chunk
    min_chunk_size : int, default=50
        Minimum tokens for initial splitting
    overlap : int, default=0
        Token overlap between final chunks

    Returns:
    --------
    Tuple[List[str], List[List[float]]]
        (text_chunks, embeddings)

    Example:
    --------
    >>> chunks, embeddings = cluster_semantic_chunking(
    ...     "research_paper.pdf",
    ...     max_chunk_size=400,
    ...     overlap=50
    ... )
    """
    # Read file
    print(f"Reading file: {file_path}")
    content = read_file(file_path)
    text = content['text']
    print(f"Extracted {len(text)} characters")

    # Step 1: Split into small sentences using internal helper
    print(f"🔄 Splitting into small chunks...")
    small_chunks = _recursive_split_text(
        text=text,
        chunk_size=min_chunk_size,
        overlap=0,
        separators=["\n\n", "\n", ".", "?", "!", " ", ""]
    )

    if len(small_chunks) == 0:
        return [], []

    print(f"Created {len(small_chunks)} small chunks")

    # Step 2: Get embeddings for small chunks
    print(f"Getting embeddings for small chunks...")
    small_embeddings = get_gemini_embeddings(small_chunks)

    # Step 3: Calculate similarity matrix
    embedding_matrix = np.array(small_embeddings)

    # Normalize vectors
    norms = np.linalg.norm(embedding_matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1  # Avoid division by zero
    embedding_matrix = embedding_matrix / norms

    similarity_matrix = np.dot(embedding_matrix, embedding_matrix.T)

    # Step 4: Dynamic programming to find optimal segmentation
    print(f"🧮 Computing optimal segmentation...")
    max_cluster = max(1, max_chunk_size // min_chunk_size)

    mean_value = np.mean(similarity_matrix[np.triu_indices(similarity_matrix.shape[0], k=1)])
    normalized_matrix = similarity_matrix - mean_value
    np.fill_diagonal(normalized_matrix, 0)

    n = len(small_chunks)
    dp = np.zeros(n)
    segmentation = np.zeros(n, dtype=int)

    for i in range(n):
        for size in range(1, min(max_cluster + 1, i + 2)):
            if i - size + 1 >= 0:
                start = i - size + 1
                end = i

                # Calculate reward for this segment
                sub_matrix = normalized_matrix[start:end+1, start:end+1]
                reward = np.sum(sub_matrix)

                adjusted_reward = reward
                if i - size >= 0:
                    adjusted_reward += dp[i - size]

                if adjusted_reward > dp[i]:
                    dp[i] = adjusted_reward
                    segmentation[i] = start

    # Step 5: Extract clusters
    clusters = []
    i = n - 1
    while i >= 0:
        start = segmentation[i]
        clusters.append((start, i))
        i = start - 1

    clusters.reverse()

    # Step 6: Merge small chunks into final chunks
    final_chunks = []
    for start, end in clusters:
        chunk_text = ' '.join(small_chunks[start:end+1])
        final_chunks.append(chunk_text)

    # Step 7: Apply overlap if specified
    if overlap > 0:
        overlapped_chunks = []
        encoding = tiktoken.get_encoding("cl100k_base")

        for i, chunk in enumerate(final_chunks):
            if i == 0:
                overlapped_chunks.append(chunk)
            else:
                # Add overlap from previous chunk (extract from string, not tokens)
                prev_chunk = final_chunks[i-1]
                prev_tokens = encoding.encode(prev_chunk)

                if len(prev_tokens) > overlap:
                    # Find the character position for last 'overlap' tokens
                    char_pos = len(prev_chunk)
                    estimate_start = max(0, char_pos - (overlap * 4))

                    overlap_text = prev_chunk[estimate_start:]
                    overlap_token_count = len(encoding.encode(overlap_text))

                    # Adjust to get exactly 'overlap' tokens
                    while overlap_token_count < overlap and estimate_start > 0:
                        estimate_start -= 20
                        overlap_text = prev_chunk[estimate_start:]
                        overlap_token_count = len(encoding.encode(overlap_text))

                    while overlap_token_count > overlap and estimate_start < char_pos - 10:
                        estimate_start += 10
                        overlap_text = prev_chunk[estimate_start:]
                        overlap_token_count = len(encoding.encode(overlap_text))

                    # Snap to word boundary at the start
                    first_space = overlap_text.find(' ')
                    if first_space > 0 and first_space < len(overlap_text) // 4:
                        overlap_text = overlap_text[first_space + 1:]

                    # Combine with space separator
                    overlapped_chunks.append(overlap_text + " " + chunk)
                else:
                    overlapped_chunks.append(chunk)

        final_chunks = overlapped_chunks

    # Step 8: Generate embeddings for final chunks (optional)
    if generate_embeddings:
        print(f"Generating embeddings for {len(final_chunks)} final chunks...")
        final_embeddings = get_gemini_embeddings(final_chunks)
        print(f"Complete! Created {len(final_chunks)} semantic chunks with embeddings")
    else:
        final_embeddings = []
        print(f"Complete! Created {len(final_chunks)} semantic chunks")

    return final_chunks, final_embeddings


# ============================================================================
# 4. KAMRADT SEMANTIC CHUNKING
# ============================================================================

def kamradt_semantic_chunking(
    file_path: str,
    avg_chunk_size: int = 400,
    min_chunk_size: int = 50,
    overlap: int = 0,
    buffer_size: int = 3,
    generate_embeddings: bool = False
) -> Tuple[List[str], List[List[float]]]:
    """
    Split document based on cosine similarity between sentence embeddings.

    Uses cosine distances to find semantic boundaries and binary search
    to achieve target average chunk size.

    Parameters:
    -----------
    file_path : str
        Path to the document file (.pdf, .docx, .txt, .md)
    avg_chunk_size : int, default=400
        Target average chunk size in tokens
    min_chunk_size : int, default=50
        Minimum size for initial sentence splitting
    overlap : int, default=0
        Token overlap between chunks
    buffer_size : int, default=3
        Number of surrounding sentences to include for context

    Returns:
    --------
    Tuple[List[str], List[List[float]]]
        (text_chunks, embeddings)

    Example:
    --------
    >>> chunks, embeddings = kamradt_semantic_chunking(
    ...     "article.txt",
    ...     avg_chunk_size=300,
    ...     overlap=30
    ... )
    """
    # Read file
    print(f"Reading file: {file_path}")
    content = read_file(file_path)
    text = content['text']
    print(f"Extracted {len(text)} characters")

    # Step 1: Split into sentences using internal helper
    sentences_list = _recursive_split_text(
        text=text,
        chunk_size=min_chunk_size,
        overlap=0,
        separators=["\n\n", "\n", ".", "?", "!", " ", ""]
    )

    if len(sentences_list) == 0:
        return [], []

    if len(sentences_list) == 1:
        embeddings = get_gemini_embeddings(sentences_list)
        return sentences_list, embeddings

    print(f"Created {len(sentences_list)} sentences")

    # Step 2: Create sentence dictionaries
    sentences = [{'sentence': s, 'index': i} for i, s in enumerate(sentences_list)]

    # Step 3: Combine sentences with buffer
    for i in range(len(sentences)):
        combined = ''

        # Add sentences before
        for j in range(i - buffer_size, i):
            if j >= 0:
                combined += sentences[j]['sentence'] + ' '

        # Add current sentence
        combined += sentences[i]['sentence']

        # Add sentences after
        for j in range(i + 1, i + 1 + buffer_size):
            if j < len(sentences):
                combined += ' ' + sentences[j]['sentence']

        sentences[i]['combined_sentence'] = combined

    # Step 4: Get embeddings for combined sentences
    combined_sentences = [s['combined_sentence'] for s in sentences]
    print(f"Getting embeddings for {len(combined_sentences)} combined sentences...")
    embeddings_list = get_gemini_embeddings(combined_sentences)

    # Step 5: Calculate cosine distances
    embedding_matrix = np.array(embeddings_list)

    # Normalize
    norms = np.linalg.norm(embedding_matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1
    embedding_matrix = embedding_matrix / norms

    similarity_matrix = np.dot(embedding_matrix, embedding_matrix.T)

    distances = []
    for i in range(len(sentences) - 1):
        similarity = similarity_matrix[i, i + 1]
        distance = 1 - similarity
        distances.append(distance)
        sentences[i]['distance_to_next'] = distance

    # Step 6: Binary search for threshold
    total_tokens = sum(openai_token_count(s['sentence']) for s in sentences)
    number_of_cuts = max(1, total_tokens // avg_chunk_size)

    print(f"Finding optimal split threshold...")
    lower_limit = 0.0
    upper_limit = 1.0
    distances_np = np.array(distances)

    while upper_limit - lower_limit > 1e-6:
        threshold = (upper_limit + lower_limit) / 2.0
        num_above = np.sum(distances_np > threshold)

        if num_above > number_of_cuts:
            lower_limit = threshold
        else:
            upper_limit = threshold

    # Step 7: Find split points
    indices_above_thresh = [i for i, x in enumerate(distances) if x > threshold]

    # Step 8: Create chunks
    start_index = 0
    chunks = []

    for index in indices_above_thresh:
        group = sentences[start_index:index + 1]
        combined_text = ' '.join([d['sentence'] for d in group])
        chunks.append(combined_text)
        start_index = index + 1

    # Add last group
    if start_index < len(sentences):
        combined_text = ' '.join([d['sentence'] for d in sentences[start_index:]])
        chunks.append(combined_text)

    # Step 9: Apply overlap if specified
    if overlap > 0 and len(chunks) > 1:
        overlapped_chunks = []
        encoding = tiktoken.get_encoding("cl100k_base")

        for i, chunk in enumerate(chunks):
            if i == 0:
                overlapped_chunks.append(chunk)
            else:
                # Add overlap from previous chunk (extract from string, not tokens)
                prev_chunk = chunks[i-1]
                prev_tokens = encoding.encode(prev_chunk)

                if len(prev_tokens) > overlap:
                    # Find the character position for last 'overlap' tokens
                    char_pos = len(prev_chunk)
                    estimate_start = max(0, char_pos - (overlap * 4))

                    overlap_text = prev_chunk[estimate_start:]
                    overlap_token_count = len(encoding.encode(overlap_text))

                    # Adjust to get exactly 'overlap' tokens
                    while overlap_token_count < overlap and estimate_start > 0:
                        estimate_start -= 20
                        overlap_text = prev_chunk[estimate_start:]
                        overlap_token_count = len(encoding.encode(overlap_text))

                    while overlap_token_count > overlap and estimate_start < char_pos - 10:
                        estimate_start += 10
                        overlap_text = prev_chunk[estimate_start:]
                        overlap_token_count = len(encoding.encode(overlap_text))

                    # Snap to word boundary at the start
                    first_space = overlap_text.find(' ')
                    if first_space > 0 and first_space < len(overlap_text) // 4:
                        overlap_text = overlap_text[first_space + 1:]

                    # Combine with space separator
                    overlapped_chunks.append(overlap_text + " " + chunk)
                else:
                    overlapped_chunks.append(chunk)

        chunks = overlapped_chunks

    # Step 10: Generate embeddings for final chunks (optional)
    if generate_embeddings:
        print(f"Generating embeddings for {len(chunks)} final chunks...")
        final_embeddings = get_gemini_embeddings(chunks)
        print(f"Complete! Created {len(chunks)} semantic chunks with embeddings")
    else:
        final_embeddings = []
        print(f"Complete! Created {len(chunks)} semantic chunks")

    return chunks, final_embeddings


# ============================================================================
# 5. LLM SEMANTIC CHUNKING
# ============================================================================

def llm_semantic_chunking(
    file_path: str,
    chunk_size: int = 400,
    overlap: int = 0,
    model_name: str = "gemini-2.5-flash",
    temperature: float = 0.2,
    generate_embeddings: bool = False
) -> Tuple[List[str], List[List[float]]]:
    """
    Split document using Gemini LLM to determine semantic boundaries.

    Uses Google Gemini to intelligently identify where text should be
    split based on thematic consistency.

    Parameters:
    -----------
    file_path : str
        Path to the document file (.pdf, .docx, .txt, .md)
    chunk_size : int, default=400
        Target chunk size (used for initial splitting)
    overlap : int, default=0
        Token overlap between chunks
    model_name : str, default="gemini-2.5-flash"
        Gemini model to use (gemini-2.5-flash, gemini-2.5-pro, gemini-2.0-flash)
    temperature : float, default=0.2
        LLM temperature for consistency
    generate_embeddings : bool, default=False
        Whether to generate embeddings (can be slow/expensive)

    Returns:
    --------
    Tuple[List[str], List[List[float]]]
        (text_chunks, embeddings)

    Example:
    --------
    >>> chunks, embeddings = llm_semantic_chunking(
    ...     "book_chapter.pdf",
    ...     overlap=50,
    ...     generate_embeddings=True
    ... )
    """
    # Read file
    print(f"Reading file: {file_path}")
    content = read_file(file_path)
    text = content['text']
    print(f"Extracted {len(text)} characters")

    # Initialize model
    model = genai.GenerativeModel(model_name)

    # Step 1: Split into small chunks for LLM analysis using internal helper
    small_chunks = _recursive_split_text(
        text=text,
        chunk_size=50,
        overlap=0,
        separators=["\n\n", "\n", ".", "?", "!", " ", ""]
    )

    if len(small_chunks) == 0:
        return [], []

    if len(small_chunks) <= 3:
        # Too few chunks, return as is
        if generate_embeddings:
            embeddings = get_gemini_embeddings(small_chunks)
            print(f"Complete! Created {len(small_chunks)} chunk(s) with embeddings")
        else:
            embeddings = []
            print(f"Complete! Created {len(small_chunks)} chunk(s)")
        return small_chunks, embeddings

    print(f"Created {len(small_chunks)} small chunks for LLM analysis")

    # Step 2: Use LLM to find split points
    split_indices = []
    current_chunk = 0

    print(f"Using Gemini LLM to analyze semantic boundaries...")

    while current_chunk < len(small_chunks) - 3:
        # Prepare chunked input for LLM
        token_count = 0
        chunked_input = ''

        for i in range(current_chunk, len(small_chunks)):
            token_count += openai_token_count(small_chunks[i])
            chunked_input += f"<|start_chunk_{i+1}|>{small_chunks[i]}<|end_chunk_{i+1}|>"

            if token_count > 800:  # Limit for LLM context
                break

        # Prepare prompt
        prompt = f"""You are an assistant specialized in splitting text into thematically consistent sections.

The text has been divided into chunks, each marked with <|start_chunk_X|> and <|end_chunk_X|> tags, where X is the chunk number.

Your task is to identify the points where splits should occur, such that consecutive chunks of similar themes stay together.

Respond with a list of chunk IDs where you believe a split should be made. For example, if chunks 1 and 2 belong together but chunk 3 starts a new topic, you would suggest a split after chunk 2.

THE CHUNKS MUST BE IN ASCENDING ORDER and must be equal to or larger than {current_chunk}.

CHUNKED_TEXT:
{chunked_input}

Respond ONLY with the format: split_after: 3, 5, 7
You MUST suggest at least ONE split."""

        # Query LLM
        max_retries = 3
        numbers = []  # Initialize to avoid undefined variable error
        for attempt in range(max_retries):
            try:
                response = model.generate_content(
                    prompt,
                    generation_config=genai.types.GenerationConfig(
                        temperature=temperature,
                        max_output_tokens=200,
                    )
                )

                result_text = response.text

                # Parse response
                split_after_line = [line for line in result_text.split('\n')
                                   if 'split_after:' in line.lower()]

                if not split_after_line:
                    # Try to find any numbers
                    numbers = re.findall(r'\d+', result_text)
                else:
                    numbers = re.findall(r'\d+', split_after_line[0])

                numbers = list(map(int, numbers))

                # Validate
                if numbers and numbers == sorted(numbers) and all(n >= current_chunk for n in numbers):
                    split_indices.extend(numbers)
                    current_chunk = numbers[-1]
                    break
                else:
                    if attempt == max_retries - 1:
                        # Give up on this batch, move forward
                        current_chunk += 5
                    time.sleep(1)

            except Exception as e:
                print(f"LLM call failed (attempt {attempt + 1}): {e}")
                if attempt == max_retries - 1:
                    current_chunk += 5
                time.sleep(2)

        if not numbers:
            break

    # Step 3: Create final chunks based on split indices
    if not split_indices:
        # No splits found, return all as one chunk
        full_text = ' '.join(small_chunks)
        if generate_embeddings:
            embeddings = get_gemini_embeddings([full_text])
            print(f"Complete! Created 1 chunk with embeddings")
        else:
            embeddings = []
            print(f"Complete! Created 1 chunk")
        return [full_text], embeddings

    chunks_to_split_after = [i - 1 for i in split_indices]

    final_chunks = []
    current_chunk_text = ''

    for i, chunk in enumerate(small_chunks):
        current_chunk_text += chunk + ' '
        if i in chunks_to_split_after:
            final_chunks.append(current_chunk_text.strip())
            current_chunk_text = ''

    if current_chunk_text:
        final_chunks.append(current_chunk_text.strip())

    # Step 4: Apply overlap if specified
    if overlap > 0 and len(final_chunks) > 1:
        overlapped_chunks = []
        encoding = tiktoken.get_encoding("cl100k_base")

        for i, chunk in enumerate(final_chunks):
            if i == 0:
                overlapped_chunks.append(chunk)
            else:
                # Add overlap from previous chunk (extract from string, not tokens)
                prev_chunk = final_chunks[i-1]
                prev_tokens = encoding.encode(prev_chunk)

                if len(prev_tokens) > overlap:
                    # Find the character position for last 'overlap' tokens
                    char_pos = len(prev_chunk)
                    estimate_start = max(0, char_pos - (overlap * 4))

                    overlap_text = prev_chunk[estimate_start:]
                    overlap_token_count = len(encoding.encode(overlap_text))

                    # Adjust to get exactly 'overlap' tokens
                    while overlap_token_count < overlap and estimate_start > 0:
                        estimate_start -= 20
                        overlap_text = prev_chunk[estimate_start:]
                        overlap_token_count = len(encoding.encode(overlap_text))

                    while overlap_token_count > overlap and estimate_start < char_pos - 10:
                        estimate_start += 10
                        overlap_text = prev_chunk[estimate_start:]
                        overlap_token_count = len(encoding.encode(overlap_text))

                    # Snap to word boundary at the start
                    first_space = overlap_text.find(' ')
                    if first_space > 0 and first_space < len(overlap_text) // 4:
                        overlap_text = overlap_text[first_space + 1:]

                    # Combine with space separator
                    overlapped_chunks.append(overlap_text + " " + chunk)
                else:
                    overlapped_chunks.append(chunk)

        final_chunks = overlapped_chunks

    # Step 5: Generate embeddings (optional)
    if generate_embeddings:
        print(f"Generating embeddings for {len(final_chunks)} final chunks...")
        final_embeddings = get_gemini_embeddings(final_chunks)
        print(f"Complete! Created {len(final_chunks)} LLM-guided chunks with embeddings")
    else:
        final_embeddings = []
        print(f"Complete! Created {len(final_chunks)} LLM-guided chunks")

    return final_chunks, final_embeddings


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    # Sample text
    sample_text = """
    Artificial Intelligence (AI) is revolutionizing technology.
    Machine learning enables computers to learn from data.
    Deep learning uses neural networks with multiple layers.

    Python is the leading language for AI development.
    Libraries like TensorFlow and PyTorch are very popular.
    These tools have democratized AI development.

    The future of AI includes natural language processing.
    Computer vision and robotics are advancing rapidly.
    Ethical considerations around AI remain important.
    """

    print("=" * 80)
    print("TEXT CHUNKING WITH GOOGLE GEMINI")
    print("=" * 80)

    # NOTE: Set your GOOGLE_API_KEY environment variable or pass it directly
    api_key = os.getenv("GOOGLE_API_KEY")

    if not api_key:
        print("\n⚠️  WARNING: GOOGLE_API_KEY not found in environment variables.")
        print("Set it with: export GOOGLE_API_KEY='your-api-key'")
        print("\nOr pass api_key parameter directly to functions.")
    else:
        print("\n✅ Google API Key found!")

        # Test Fixed Token Chunking
        print("\n" + "="*80)
        print("1. FIXED TOKEN CHUNKING")
        print("="*80)
        chunks, embeddings = fixed_token_chunking(
            sample_text,
            chunk_size=100,
            overlap=20,
            api_key=api_key
        )
        print(f"Created {len(chunks)} chunks")
        print(f"Embedding dimension: {len(embeddings[0])}")
        print(f"First chunk: {chunks[0][:80]}...")

        # Test Recursive Token Chunking
        print("\n" + "="*80)
        print("2. RECURSIVE TOKEN CHUNKING")
        print("="*80)
        chunks, embeddings = recursive_token_chunking(
            sample_text,
            chunk_size=150,
            overlap=20,
            api_key=api_key
        )
        print(f"Created {len(chunks)} chunks")
        for i, chunk in enumerate(chunks[:3], 1):
            print(f"Chunk {i}: {chunk[:80]}...")
