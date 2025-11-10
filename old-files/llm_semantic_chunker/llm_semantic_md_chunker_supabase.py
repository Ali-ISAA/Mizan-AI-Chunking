"""
LLM Semantic Markdown Chunker - SUPABASE VERSION
=================================================

Uses Google Gemini LLM to intelligently chunk markdown files based PURELY on semantic meaning.
The LLM decides ALL chunk boundaries - no size limits!

This version saves embeddings to SUPABASE (PostgreSQL with pgvector) instead of ChromaDB.

Features:
- Extracts markdown sections (headers)
- Uses Gemini LLM to determine ALL split points based on meaning
- LLM analyzes thematic shifts and decides boundaries
- No artificial size constraints - chunks can be any size
- Maintains parent section context
- Saves to Supabase with pgvector for semantic search

Strategy:
1. Extract sections from markdown
2. For each section, let LLM decide ALL split points
3. LLM groups semantically related sentences together
4. Chunks are created based purely on meaning, not size
5. Save to Supabase with embeddings
"""

import re
import sys
import os
import tiktoken
import google.generativeai as genai
import chromadb
import uuid
import json
from typing import List, Dict, Tuple
from pathlib import Path
from dotenv import load_dotenv
from supabase import create_client, Client

# Load environment variables
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

# Add parent directory to path to import from other modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from other_chunkers.text_chunking_methods import GEMINI_API_KEYS, get_gemini_embeddings
from llm_semantic_chunker.api_key_manager import initialize_key_manager, get_key_manager

# Initialize API key manager for LLM semantic chunking
_key_manager = initialize_key_manager(GEMINI_API_KEYS)


class LLMSemanticMarkdownChunker:
    """
    Intelligent markdown chunker using LLM for semantic analysis.
    """

    def __init__(
        self,
        md_file_path: str,
        model_name: str = "gemini-2.0-flash-lite",
        temperature: float = 0.2
    ):
        """
        Initialize the LLM semantic markdown chunker.

        Parameters:
        -----------
        md_file_path : str
            Path to the markdown (.md) file
        model_name : str
            Gemini model to use (default: gemini-2.0-flash-lite)
        temperature : float
            LLM temperature for consistency (default: 0.2)
        """
        self.md_file_path = md_file_path
        self.file_name = Path(md_file_path).stem
        self.encoding = tiktoken.get_encoding("cl100k_base")
        self.content = self._read_file()

        # Initialize Gemini model
        self.model = genai.GenerativeModel(
            model_name,
            generation_config=genai.types.GenerationConfig(
                temperature=temperature,
                max_output_tokens=500,
            )
        )

    def _read_file(self) -> str:
        """Read the markdown file content."""
        with open(self.md_file_path, 'r', encoding='utf-8') as f:
            return f.read()

    def _count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        return len(self.encoding.encode(text))

    def _clean_content(self, text: str) -> str:
        """Clean content by removing image placeholders and extra whitespace."""
        # Remove image placeholders
        text = re.sub(r'<!-- image -->\s*', '', text)
        # Remove multiple consecutive newlines
        text = re.sub(r'\n{3,}', '\n\n', text)
        # Remove leading/trailing whitespace
        text = text.strip()
        return text

    def _extract_sections(self) -> List[Dict[str, str]]:
        """Extract sections from markdown based on headers."""
        sections = []
        lines = self.content.split('\n')

        current_section = {
            'level': 0,
            'title': 'Document Header',
            'content': [],
            'parent': None
        }

        parent_h2 = None

        for line in lines:
            # Check for H2 header (##)
            h2_match = re.match(r'^##\s+(.+)$', line)
            if h2_match:
                if current_section['content']:
                    section_text = '\n'.join(current_section['content']).strip()
                    if section_text:
                        current_section['content'] = section_text
                        sections.append(current_section.copy())

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
                if current_section['content']:
                    section_text = '\n'.join(current_section['content']).strip()
                    if section_text:
                        current_section['content'] = section_text
                        sections.append(current_section.copy())

                title = h3_match.group(1).strip()
                current_section = {
                    'level': 3,
                    'title': title,
                    'content': [],
                    'parent': parent_h2
                }
                continue

            current_section['content'].append(line)

        # Last section
        if current_section['content']:
            section_text = '\n'.join(current_section['content']).strip()
            if section_text:
                current_section['content'] = section_text
                sections.append(current_section.copy())

        return sections

    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        # Split by sentence-ending punctuation followed by space or newline
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

        # Create prompt for markdown-specific analysis
        prompt = f"""You are analyzing a markdown section titled "{section_title}" from a structured document.

The text has been split into sentences, each marked with <|sent_X|> and <|/sent_X|> tags.

Your task: Identify where COMPLETE TOPIC BOUNDARIES occur. Each chunk should contain:
- A main heading/topic with ALL its related content
- Complete explanations, lists, and details about that topic
- NO splitting within a single concept or explanation

Guidelines:
- Keep related sentences that explain the SAME heading/topic together
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
            # Get key manager and execute with retry
            manager = get_key_manager()

            def llm_call():
                response = self.model.generate_content(prompt)
                return response.text

            result_text = manager.execute_with_retry(llm_call)

            # Parse response
            if 'none' in result_text.lower():
                return []

            split_line = [line for line in result_text.split('\n')
                         if 'split_after:' in line.lower()]

            if split_line:
                # Extract numbers
                numbers_str = split_line[0].split('split_after:')[1]
                numbers = [int(n.strip()) for n in numbers_str.split(',') if n.strip().isdigit()]
                # Filter valid indices
                return [n for n in numbers if 0 <= n < len(sentences) - 1]

            return []

        except Exception as e:
            print(f"⚠️  LLM split analysis failed: {e}")
            return []

    def _chunk_section_with_llm(self, section: Dict) -> List[Tuple[str, Dict]]:
        """
        Chunk a section using PURE LLM semantic analysis - no size limits!

        Parameters:
        -----------
        section : Dict
            Section dictionary

        Returns:
        --------
        List[Tuple[str, Dict]]
            List of (chunk_text, metadata) tuples
        """
        content = section['content']

        # Build context header
        context_parts = []
        if section['parent']:
            context_parts.append(f"[Parent Section: {section['parent']}]")

        if section['level'] == 2:
            context_parts.append(f"## {section['title']}")
        elif section['level'] == 3:
            context_parts.append(f"### {section['title']}")

        context_header = '\n'.join(context_parts)

        # Always use LLM to analyze - no size check!
        print(f"  🤖 Using LLM to analyze: {section['title']}")

        sentences = self._split_into_sentences(content)

        if len(sentences) <= 2:
            # Too few sentences, keep as single chunk
            chunk_text = context_header + '\n\n' + content
            chunk_text = self._clean_content(chunk_text)
            return [(chunk_text, {
                'title': section['title'],
                'level': section['level'],
                'parent': section['parent'],
                'split_by_llm': False
            })]

        # Get split points from LLM - LLM decides everything!
        split_indices = self._llm_find_split_points(sentences, section['title'])

        if not split_indices:
            # LLM says no splits needed - keep entire section as one chunk
            chunk_text = context_header + '\n\n' + content
            chunk_text = self._clean_content(chunk_text)
            return [(chunk_text, {
                'title': section['title'],
                'level': section['level'],
                'parent': section['parent'],
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
                    'level': section['level'],
                    'parent': section['parent'],
                    'split_by_llm': True,
                    'chunk_part': i + 1
                }))

        return chunks if chunks else [(context_header + '\n\n' + content, {
            'title': section['title'],
            'level': section['level'],
            'parent': section['parent'],
            'split_by_llm': True,
            'llm_decision': 'fallback'
        })]

    def _split_by_size(self, section: Dict, context_header: str) -> List[Tuple[str, Dict]]:
        """Fallback: split by size when LLM doesn't suggest splits."""
        chunks = []
        content = section['content']
        paragraphs = re.split(r'\n\n+', content)

        current_chunk = []
        current_tokens = self._count_tokens(context_header)

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            para_tokens = self._count_tokens(para)

            if current_tokens + para_tokens > self.target_chunk_size and current_chunk:
                chunk_text = context_header + '\n\n' + '\n\n'.join(current_chunk)
                chunk_text = self._clean_content(chunk_text)
                chunks.append((chunk_text, {
                    'title': section['title'],
                    'level': section['level'],
                    'parent': section['parent'],
                    'split_by_llm': False,
                    'chunk_part': len(chunks) + 1
                }))
                current_chunk = [para]
                current_tokens = self._count_tokens(context_header + '\n\n' + para)
            else:
                current_chunk.append(para)
                current_tokens += para_tokens

        if current_chunk:
            chunk_text = context_header + '\n\n' + '\n\n'.join(current_chunk)
            chunk_text = self._clean_content(chunk_text)
            chunks.append((chunk_text, {
                'title': section['title'],
                'level': section['level'],
                'parent': section['parent'],
                'split_by_llm': False,
                'chunk_part': len(chunks) + 1
            }))

        return chunks

    def chunk_with_llm_semantics(self) -> Tuple[List[str], List[Dict]]:
        """
        Chunk markdown using PURE LLM semantic analysis - no size limits!

        Returns:
        --------
        Tuple[List[str], List[Dict]]
            - List of chunk texts
            - List of chunk metadata
        """
        print(f"\n📄 Processing: {self.md_file_path}")
        print(f"🎯 LLM decides ALL boundaries - no size limits!\n")

        sections = self._extract_sections()
        print(f"✓ Extracted {len(sections)} sections\n")

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

        # Post-process: split any chunks that exceed safe size limit
        print(f"\n🔍 Checking for oversized chunks (> 800 tokens / 16KB)...")
        final_chunks, final_metadata = self._split_oversized_chunks(chunks, metadata)

        if len(final_chunks) > len(chunks):
            print(f"✓ Split {len(final_chunks) - len(chunks)} oversized chunks into smaller semantic pieces")

        return final_chunks, final_metadata

    def _split_oversized_chunks(self, chunks: List[str], metadata: List[Dict]) -> Tuple[List[str], List[Dict]]:
        """
        Split chunks that exceed safe size limits while maintaining semantic boundaries.

        Parameters:
        -----------
        chunks : List[str]
            Original chunks
        metadata : List[Dict]
            Original metadata

        Returns:
        --------
        Tuple[List[str], List[Dict]]
            Processed chunks and metadata with oversized chunks split
        """
        MAX_SAFE_TOKENS = 800  # Safe limit well below 16KB
        MAX_SAFE_BYTES = 14000  # Safe byte limit (slightly below 16KB)

        final_chunks = []
        final_metadata = []

        for chunk, meta in zip(chunks, metadata):
            chunk_tokens = self._count_tokens(chunk)
            chunk_bytes = len(chunk.encode('utf-8'))

            # If chunk is within safe limits, keep as is
            if chunk_tokens <= MAX_SAFE_TOKENS and chunk_bytes <= MAX_SAFE_BYTES:
                final_chunks.append(chunk)
                final_metadata.append(meta)
                continue

            # Chunk is too large - split it semantically
            print(f"  ℹ️  Splitting large chunk: '{meta['title']}' ({chunk_tokens} tokens, {chunk_bytes} bytes)")

            # Extract header and content
            lines = chunk.split('\n')
            header_lines = []
            content_start = 0

            for i, line in enumerate(lines):
                if line.startswith('[Parent Section:') or line.startswith('##') or line.startswith('###'):
                    header_lines.append(line)
                    content_start = i + 1
                elif header_lines and not line.strip():
                    content_start = i + 1
                    break

            header = '\n'.join(header_lines)
            content = '\n'.join(lines[content_start:])

            # Split content by paragraphs
            paragraphs = re.split(r'\n\n+', content)

            current_sub_chunk = []
            current_tokens = self._count_tokens(header)
            sub_chunk_num = 1

            for para in paragraphs:
                para = para.strip()
                if not para:
                    continue

                para_tokens = self._count_tokens(para)

                # If single paragraph exceeds limit, split by sentences
                if para_tokens > MAX_SAFE_TOKENS:
                    # Save current sub-chunk first
                    if current_sub_chunk:
                        sub_text = header + '\n\n' + '\n\n'.join(current_sub_chunk)
                        sub_text = self._clean_content(sub_text)
                        final_chunks.append(sub_text)
                        final_metadata.append({
                            **meta,
                            'tokens': self._count_tokens(sub_text),
                            'sub_chunk_part': sub_chunk_num,
                            'split_reason': 'size_exceeded'
                        })
                        sub_chunk_num += 1
                        current_sub_chunk = []
                        current_tokens = self._count_tokens(header)

                    # Split paragraph by sentences
                    sentences = re.split(r'(?<=[.!?])\s+', para)
                    sentence_group = []
                    sentence_tokens = self._count_tokens(header)

                    for sentence in sentences:
                        sent_tokens = self._count_tokens(sentence)

                        if sentence_tokens + sent_tokens > MAX_SAFE_TOKENS:
                            if sentence_group:
                                sub_text = header + '\n\n' + ' '.join(sentence_group)
                                sub_text = self._clean_content(sub_text)
                                final_chunks.append(sub_text)
                                final_metadata.append({
                                    **meta,
                                    'tokens': self._count_tokens(sub_text),
                                    'sub_chunk_part': sub_chunk_num,
                                    'split_reason': 'size_exceeded'
                                })
                                sub_chunk_num += 1
                            sentence_group = [sentence]
                            sentence_tokens = self._count_tokens(header + '\n\n' + sentence)
                        else:
                            sentence_group.append(sentence)
                            sentence_tokens += sent_tokens

                    # Add remaining sentences
                    if sentence_group:
                        sub_text = header + '\n\n' + ' '.join(sentence_group)
                        sub_text = self._clean_content(sub_text)
                        final_chunks.append(sub_text)
                        final_metadata.append({
                            **meta,
                            'tokens': self._count_tokens(sub_text),
                            'sub_chunk_part': sub_chunk_num,
                            'split_reason': 'size_exceeded'
                        })
                        sub_chunk_num += 1
                        current_sub_chunk = []
                        current_tokens = self._count_tokens(header)

                # If adding paragraph would exceed limit, save current sub-chunk
                elif current_tokens + para_tokens > MAX_SAFE_TOKENS:
                    if current_sub_chunk:
                        sub_text = header + '\n\n' + '\n\n'.join(current_sub_chunk)
                        sub_text = self._clean_content(sub_text)
                        final_chunks.append(sub_text)
                        final_metadata.append({
                            **meta,
                            'tokens': self._count_tokens(sub_text),
                            'sub_chunk_part': sub_chunk_num,
                            'split_reason': 'size_exceeded'
                        })
                        sub_chunk_num += 1
                    current_sub_chunk = [para]
                    current_tokens = self._count_tokens(header + '\n\n' + para)
                else:
                    current_sub_chunk.append(para)
                    current_tokens += para_tokens

            # Add remaining sub-chunk
            if current_sub_chunk:
                sub_text = header + '\n\n' + '\n\n'.join(current_sub_chunk)
                sub_text = self._clean_content(sub_text)
                final_chunks.append(sub_text)
                final_metadata.append({
                    **meta,
                    'tokens': self._count_tokens(sub_text),
                    'sub_chunk_part': sub_chunk_num,
                    'split_reason': 'size_exceeded'
                })

        return final_chunks, final_metadata

    def generate_embeddings_and_save(self, chunks: List[str] = None, metadata: List[Dict] = None, collection_name: str = None):
        """
        Generate embeddings for chunks and save to ChromaDB with metadata.

        Parameters:
        -----------
        chunks : List[str], optional
            Pre-computed chunks. If None, will call chunk_with_llm_semantics()
        metadata : List[Dict], optional
            Pre-computed metadata for chunks
        collection_name : str, optional
            Name for the ChromaDB collection. If None, uses file name.
        """
        print("\n" + "=" * 80)
        print("GENERATING EMBEDDINGS AND SAVING TO CHROMADB")
        print("=" * 80)

        # Get chunks and metadata if not provided
        if chunks is None or metadata is None:
            chunks, metadata = self.chunk_with_llm_semantics()

        if not chunks:
            print("\n⚠️  No chunks to save!")
            return

        # Filter out very short chunks (less than 50 characters)
        filtered_chunks = []
        filtered_metadata = []

        for chunk, meta in zip(chunks, metadata):
            if len(chunk) >= 50:
                filtered_chunks.append(chunk)
                filtered_metadata.append(meta)

        if not filtered_chunks:
            print("\n⚠️  No valid chunks after filtering!")
            return

        print(f"\n✓ Total chunks after filtering: {len(filtered_chunks)}")
        print(f"  (Filtered out {len(chunks) - len(filtered_chunks)} chunks < 50 chars)")

        # Generate embeddings
        print(f"\n🔄 Generating embeddings for {len(filtered_chunks)} chunks...")
        embeddings = get_gemini_embeddings(filtered_chunks)

        if not embeddings or len(embeddings) != len(filtered_chunks):
            print(f"\n❌ Embedding generation failed!")
            return

        print(f"✓ Generated {len(embeddings)} embeddings")

        # Set collection name
        if collection_name is None:
            # Sanitize file name for ChromaDB (only alphanumeric, dots, underscores, hyphens)
            sanitized_name = re.sub(r'[^a-zA-Z0-9._-]', '_', self.file_name)
            # Remove consecutive underscores
            sanitized_name = re.sub(r'_+', '_', sanitized_name)
            # Remove leading/trailing underscores or hyphens
            sanitized_name = sanitized_name.strip('_-')
            collection_name = f"llm_semantic_md_{sanitized_name}"

        # Connect to ChromaDB Cloud
        print(f"\n🔄 Connecting to ChromaDB...")
        client = chromadb.CloudClient(
            api_key=os.getenv('CHROMADB_API_KEY'),
            tenant=os.getenv('CHROMADB_TENANT'),
            database=os.getenv('CHROMADB_DATABASE')
        )
        print(f"✓ Connected to ChromaDB Cloud")

        # Create or get collection (will replace if exists)
        try:
            client.delete_collection(name=collection_name)
            print(f"✓ Deleted existing collection: {collection_name}")
        except:
            pass

        collection = client.get_or_create_collection(name=collection_name)
        print(f"✓ Created collection: {collection_name}")

        # Prepare metadata for ChromaDB
        chroma_metadata = []
        for meta in filtered_metadata:
            chroma_meta = {
                'title': str(meta['title']),
                'level': str(meta['level']),
                'parent': str(meta.get('parent', '')),
                'tokens': str(meta['tokens']),
                'split_by_llm': str(meta['split_by_llm']),
                'file': str(meta['file'])
            }

            # Add optional fields
            if meta.get('chunk_part'):
                chroma_meta['chunk_part'] = str(meta['chunk_part'])
            if meta.get('llm_decision'):
                chroma_meta['llm_decision'] = str(meta['llm_decision'])
            if meta.get('sub_chunk_part'):
                chroma_meta['sub_chunk_part'] = str(meta['sub_chunk_part'])
            if meta.get('split_reason'):
                chroma_meta['split_reason'] = str(meta['split_reason'])

            chroma_metadata.append(chroma_meta)

        # Generate unique IDs using uuid (same as example_usage.py)
        ids = [str(uuid.uuid4()) for _ in filtered_chunks]

        # Add to ChromaDB (matching example_usage.py order: ids, documents, embeddings, metadatas)
        print(f"\n🔄 Saving {len(filtered_chunks)} chunks to ChromaDB...")
        collection.add(
            ids=ids,
            documents=filtered_chunks,
            embeddings=embeddings,
            metadatas=chroma_metadata
        )

        print(f"✓ Successfully saved {len(filtered_chunks)} chunks to ChromaDB!")
        print(f"\n📊 Collection Statistics:")
        print(f"  - Collection Name: {collection_name}")
        print(f"  - Total Chunks: {len(filtered_chunks)}")
        print(f"  - LLM-Split Chunks: {sum(1 for m in filtered_metadata if m['split_by_llm'])}")
        print(f"  - Average Tokens: {sum(m['tokens'] for m in filtered_metadata) // len(filtered_metadata)}")
        print(f"  - Min Tokens: {min(m['tokens'] for m in filtered_metadata)}")
        print(f"  - Max Tokens: {max(m['tokens'] for m in filtered_metadata)}")

        return collection_name

    def generate_embeddings_and_save_to_supabase(self, chunks: List[str] = None, metadata: List[Dict] = None, table_name: str = None):
        """
        Generate embeddings for chunks and save to Supabase with metadata.

        Parameters:
        -----------
        chunks : List[str], optional
            Pre-computed chunks. If None, will call chunk_with_llm_semantics()
        metadata : List[Dict], optional
            Pre-computed metadata for chunks
        table_name : str, optional
            Name for the Supabase table. If None, uses file name.
        """
        print("\n" + "=" * 80)
        print("GENERATING EMBEDDINGS AND SAVING TO SUPABASE")
        print("=" * 80)

        # Get chunks and metadata if not provided
        if chunks is None or metadata is None:
            chunks, metadata = self.chunk_with_llm_semantics()

        if not chunks:
            print("\n⚠️  No chunks to save!")
            return

        # Filter out very short chunks (less than 50 characters)
        filtered_chunks = []
        filtered_metadata = []

        for chunk, meta in zip(chunks, metadata):
            if len(chunk) >= 50:
                filtered_chunks.append(chunk)
                filtered_metadata.append(meta)

        if not filtered_chunks:
            print("\n⚠️  No valid chunks after filtering!")
            return

        print(f"\n✓ Total chunks after filtering: {len(filtered_chunks)}")
        print(f"  (Filtered out {len(chunks) - len(filtered_chunks)} chunks < 50 chars)")

        # Generate embeddings
        print(f"\n🔄 Generating embeddings for {len(filtered_chunks)} chunks...")
        embeddings = get_gemini_embeddings(filtered_chunks)

        if not embeddings or len(embeddings) != len(filtered_chunks):
            print(f"\n❌ Embedding generation failed!")
            return

        print(f"✓ Generated {len(embeddings)} embeddings (768 dimensions)")

        # Set table name
        if table_name is None:
            # Sanitize file name for table name
            sanitized_name = re.sub(r'[^a-zA-Z0-9_]', '_', self.file_name)
            # Remove consecutive underscores
            sanitized_name = re.sub(r'_+', '_', sanitized_name)
            # Remove leading/trailing underscores
            sanitized_name = sanitized_name.strip('_').lower()
            table_name = f"doc_{sanitized_name}"

        print(f"\n📊 Table name: {table_name}")

        # Connect to Supabase
        print(f"\n🔄 Connecting to Supabase...")
        supabase_url = os.getenv('SUPABASE_URL')
        supabase_key = os.getenv('SUPABASE_SERVICE_KEY')  # Using service key for admin operations

        if not supabase_url or not supabase_key:
            print("❌ Supabase credentials not found in .env file!")
            print("   Required: SUPABASE_URL and SUPABASE_SERVICE_KEY")
            return

        supabase: Client = create_client(supabase_url, supabase_key)
        print(f"✓ Connected to Supabase")

        # Check if table exists by trying to query it
        print(f"\n🔄 Checking if table '{table_name}' exists...")
        table_exists = False
        try:
            # Try to select from table (limit 0 for quick check)
            supabase.table(table_name).select("id").limit(0).execute()
            table_exists = True
            print(f"✓ Table '{table_name}' already exists")
        except Exception as e:
            # Table doesn't exist - create it using Supabase function
            print(f"⚠️  Table '{table_name}' does not exist")
            print(f"🔄 Creating table automatically...")

            try:
                # Call the create_document_table function via RPC
                _ = supabase.rpc('create_document_table', {'table_name': table_name}).execute()
                print(f"✓ Table '{table_name}' created successfully!")

                # Small delay to ensure table is ready
                import time
                time.sleep(1)

            except Exception as create_error:
                error_msg = str(create_error)
                print(f"❌ Automatic table creation failed: {error_msg}")

                # Check if it's because the function doesn't exist
                if 'create_document_table' in error_msg or 'PGRST202' in error_msg:
                    print(f"\n{'='*80}")
                    print(f"⚠️  ONE-TIME SETUP REQUIRED")
                    print(f"{'='*80}")
                    print(f"\nYou need to run this SQL ONCE in your Supabase SQL Editor:")
                    print(f"\nFile location: llm_semantic_chunker/supabase_setup_function.sql")
                    print(f"\nOr copy this SQL:")
                    print(f"\n{'-'*80}")

                    setup_sql = """CREATE OR REPLACE FUNCTION create_document_table(table_name text)
RETURNS void LANGUAGE plpgsql SECURITY DEFINER AS $$
BEGIN
  EXECUTE format('CREATE TABLE IF NOT EXISTS %I (
    id bigserial primary key, content text,
    metadata jsonb, embedding vector(768))', table_name);
  EXECUTE format('CREATE INDEX IF NOT EXISTS %I ON %I
    USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100)',
    table_name || '_embedding_idx', table_name);
END; $$;
GRANT EXECUTE ON FUNCTION create_document_table(text) TO service_role, authenticated, anon;"""

                    print(setup_sql)
                    print(f"{'-'*80}")
                    print(f"\nAfter running this, restart the script. It will work automatically!")
                    print(f"{'='*80}\n")
                    return None
                else:
                    # Some other error
                    print(f"\n⚠️  Unexpected error. Please check your Supabase connection.")
                    return None

            # Verify table exists
            print(f"🔄 Verifying table...")
            try:
                supabase.table(table_name).select("id").limit(0).execute()
                print(f"✓ Table '{table_name}' ready!")
            except Exception as verify_error:
                print(f"❌ Table verification failed: {verify_error}")
                return None

        # Prepare data for insertion
        print(f"\n🔄 Preparing {len(filtered_chunks)} records for insertion...")
        records = []

        for i, (chunk, meta, embedding) in enumerate(zip(filtered_chunks, filtered_metadata, embeddings)):
            # Convert embedding to list for JSON serialization
            embedding_list = embedding.tolist() if hasattr(embedding, 'tolist') else list(embedding)

            # Prepare metadata as JSON
            metadata_json = {
                'title': str(meta['title']),
                'level': int(meta['level']),
                'parent': str(meta.get('parent', '')),
                'tokens': int(meta['tokens']),
                'split_by_llm': bool(meta['split_by_llm']),
                'file': str(meta['file'])
            }

            # Add optional fields
            if meta.get('chunk_part'):
                metadata_json['chunk_part'] = int(meta['chunk_part'])
            if meta.get('llm_decision'):
                metadata_json['llm_decision'] = str(meta['llm_decision'])
            if meta.get('sub_chunk_part'):
                metadata_json['sub_chunk_part'] = int(meta['sub_chunk_part'])
            if meta.get('split_reason'):
                metadata_json['split_reason'] = str(meta['split_reason'])

            record = {
                'content': chunk,
                'metadata': metadata_json,
                'embedding': embedding_list
            }

            records.append(record)

        # Insert data in batches (Supabase has a limit)
        BATCH_SIZE = 100
        print(f"\n🔄 Inserting data in batches of {BATCH_SIZE}...")

        total_inserted = 0
        for i in range(0, len(records), BATCH_SIZE):
            batch = records[i:i + BATCH_SIZE]
            try:
                _ = supabase.table(table_name).insert(batch).execute()
                total_inserted += len(batch)
                print(f"  ✓ Inserted batch {i // BATCH_SIZE + 1} ({len(batch)} records) - Total: {total_inserted}/{len(records)}")
            except Exception as e:
                print(f"  ❌ Error inserting batch {i // BATCH_SIZE + 1}: {e}")
                print(f"     Continuing with next batch...")

        if total_inserted == len(records):
            print(f"\n✅ Successfully saved all {total_inserted} chunks to Supabase!")
        else:
            print(f"\n⚠️  Inserted {total_inserted}/{len(records)} chunks (some failed)")

        print(f"\n📊 Table Statistics:")
        print(f"  - Table Name: {table_name}")
        print(f"  - Total Chunks: {total_inserted}")
        print(f"  - LLM-Split Chunks: {sum(1 for m in filtered_metadata if m['split_by_llm'])}")
        print(f"  - Average Tokens: {sum(m['tokens'] for m in filtered_metadata) // len(filtered_metadata)}")
        print(f"  - Min Tokens: {min(m['tokens'] for m in filtered_metadata)}")
        print(f"  - Max Tokens: {max(m['tokens'] for m in filtered_metadata)}")
        print(f"\n🔍 You can now query this table using vector similarity search!")
        print(f"   Use the match_documents() function with table name: {table_name}")

        return table_name

    def save_chunks_to_file(self, chunks: List[str] = None, metadata: List[Dict] = None, output_path: str = None):
        """
        Save chunks to a text file for review.

        Parameters:
        -----------
        chunks : List[str], optional
            Pre-computed chunks. If None, will call chunk_with_llm_semantics()
        metadata : List[Dict], optional
            Pre-computed metadata for chunks
        output_path : str, optional
            Path to save the output file
        """
        if output_path is None:
            # Save to output/ folder (go up one level from llm_semantic_chunker/)
            output_dir = Path(__file__).parent.parent / "output"
            output_dir.mkdir(exist_ok=True)
            output_path = str(output_dir / "output_llm_semantic_md_chunks.txt")

        # Get chunks and metadata if not provided
        if chunks is None or metadata is None:
            chunks, metadata = self.chunk_with_llm_semantics()

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("LLM SEMANTIC MARKDOWN CHUNKING RESULTS\n")
            f.write("(LLM Decides ALL Boundaries - No Size Limits!)\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Source File: {self.md_file_path}\n")
            f.write(f"Chunking Method: Pure LLM semantic analysis\n")
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
                f.write(f"Split by LLM: {'Yes' if meta['split_by_llm'] else 'No'}\n")
                if meta.get('chunk_part'):
                    f.write(f"Part: {meta['chunk_part']}\n")
                if meta.get('sub_chunk_part'):
                    f.write(f"Sub-chunk Part: {meta['sub_chunk_part']} (split due to size)\n")
                if meta.get('split_reason'):
                    f.write(f"Split Reason: {meta['split_reason']}\n")
                f.write(f"{'-'*80}\n")
                f.write(chunk)
                f.write(f"\n{'-'*80}\n")

        print(f"\n✓ Saved {len(chunks)} chunks to: {output_path}")
        print(f"\n📊 Chunking Statistics:")
        print(f"  - Total Chunks: {len(chunks)}")
        print(f"  - LLM-Split Chunks: {sum(1 for m in metadata if m['split_by_llm'])}")
        print(f"  - Size-Split Chunks: {sum(1 for m in metadata if not m['split_by_llm'])}")
        print(f"  - Average Tokens: {sum(m['tokens'] for m in metadata) // len(metadata)}")
        print(f"  - Min Tokens: {min(m['tokens'] for m in metadata)}")
        print(f"  - Max Tokens: {max(m['tokens'] for m in metadata)}")


def main():
    """Main function to demonstrate LLM semantic markdown chunking."""
    print("=" * 80)
    print("LLM SEMANTIC MARKDOWN CHUNKER")
    print("(Pure LLM-Driven - No Size Limits!)")
    print("=" * 80)
    print("\nIntelligent chunker that:")
    print("  • LLM decides ALL chunk boundaries based on meaning")
    print("  • NO artificial size constraints")
    print("  • Respects markdown structure (headers)")
    print("  • Splits ONLY at thematic shifts")
    print("  • Chunks can be any size - small or large!\n")

    # Get markdown file from user
    print("=" * 80)
    print("SELECT MARKDOWN FILE TO PROCESS")
    print("=" * 80)

    # Look for .md files in MD_FILES folder (project root / MD_FILES)
    project_root = Path(__file__).parent.parent  # Go up from llm_semantic_chunker/ to MizanAiChunking/
    md_files_dir = project_root / "MD_FILES"

    # Create MD_FILES directory if it doesn't exist
    md_files_dir.mkdir(exist_ok=True)

    # Search for .md files in MD_FILES folder
    md_files = list(md_files_dir.glob("*.md"))

    # Filter out README files and hidden files
    md_files = [f for f in md_files if f.name.lower() != 'readme.md' and not f.name.startswith('.')]
    md_files.sort()

    if not md_files:
        print(f"\n⚠️  No .md files found in: {md_files_dir}")
        print("\nPlease either:")
        print(f"1. Place your .md files in: {md_files_dir}")
        print("2. Enter the full path to your markdown file")

        choice = input("\nEnter file path (or press Enter to exit): ").strip().strip('"')

        if not choice:
            print("Exiting...")
            return

        md_file_path = choice
        if not Path(md_file_path).exists():
            print("❌ Invalid file path!")
            return
    else:
        print(f"\nFound {len(md_files)} markdown file(s) in MD_FILES folder:\n")
        for i, file in enumerate(md_files, 1):
            # Show file size
            size_mb = file.stat().st_size / (1024 * 1024)
            print(f"{i}. {file.name}")
            print(f"   Size: {size_mb:.2f} MB")

        print(f"\n{len(md_files) + 1}. Enter custom path")

        choice = input(f"\nSelect file (1-{len(md_files) + 1}): ").strip()

        try:
            choice_num = int(choice)
            if 1 <= choice_num <= len(md_files):
                md_file_path = str(md_files[choice_num - 1])
            elif choice_num == len(md_files) + 1:
                md_file_path = input("\nEnter file path: ").strip().strip('"')
                if not Path(md_file_path).exists():
                    print("❌ File not found!")
                    return
            else:
                print("❌ Invalid selection!")
                return
        except ValueError:
            print("❌ Invalid input!")
            return

    print(f"\n📄 Selected: {md_file_path}\n")

    # Create chunker
    chunker = LLMSemanticMarkdownChunker(
        md_file_path=md_file_path,
        model_name="gemini-2.0-flash-lite",
        temperature=0.2
    )

    # Get chunks - ANALYSIS DONE ONLY ONCE!
    print("🔄 Performing LLM semantic analysis (this will take some time)...\n")
    chunks, metadata = chunker.chunk_with_llm_semantics()

    # Preview first 3 chunks
    print("\n" + "=" * 80)
    print("PREVIEW: First 3 Chunks")
    print("=" * 80)

    for i, (chunk, meta) in enumerate(zip(chunks[:3], metadata[:3]), 1):
        print(f"\n--- Chunk #{i} ---")
        print(f"Title: {meta['title']}")
        print(f"Level: H{meta['level']}")
        if meta['parent']:
            print(f"Parent: {meta['parent']}")
        print(f"Tokens: {meta['tokens']}")
        print(f"Split by LLM: {'Yes' if meta['split_by_llm'] else 'No'}")
        print(f"\nContent Preview (first 300 chars):")
        print(chunk[:300] + "..." if len(chunk) > 300 else chunk)
        print("-" * 80)

    # Ask user what to do next
    print("\n" + "=" * 80)
    print("NEXT STEPS - SUPABASE VERSION")
    print("=" * 80)
    print("1. Save chunks to file")
    print("2. Generate embeddings and save to ChromaDB")
    print("3. Generate embeddings and save to SUPABASE")
    print("4. Both ChromaDB AND Supabase")
    print("5. All three (file + ChromaDB + Supabase)")
    print("6. Exit")

    next_step = input("\nSelect option (1-6): ").strip()

    if next_step == "1" or next_step == "5":
        print("\n" + "=" * 80)
        print("Saving all chunks to file...")
        print("=" * 80)
        chunker.save_chunks_to_file(chunks=chunks, metadata=metadata)

    if next_step == "2" or next_step == "4" or next_step == "5":
        print("\n" + "=" * 80)
        print("Generating embeddings and saving to ChromaDB...")
        print("=" * 80)
        collection_name = chunker.generate_embeddings_and_save(chunks=chunks, metadata=metadata)

        print("\n" + "=" * 80)
        print("✅ CHROMADB SAVE COMPLETED!")
        print("=" * 80)
        print(f"\n📦 Collection: {collection_name}")
        print("🎉 Chunks saved to ChromaDB!")

    if next_step == "3" or next_step == "4" or next_step == "5":
        print("\n" + "=" * 80)
        print("Generating embeddings and saving to SUPABASE...")
        print("=" * 80)
        table_name = chunker.generate_embeddings_and_save_to_supabase(chunks=chunks, metadata=metadata)

        print("\n" + "=" * 80)
        print("✅ SUPABASE SAVE COMPLETED!")
        print("=" * 80)
        print(f"\n📊 Table: {table_name}")
        print("🎉 Chunks saved to Supabase with vector search enabled!")

    if next_step in ["2", "3", "4", "5"]:
        print("\n" + "=" * 80)
        print("✅ ALL TASKS COMPLETED!")
        print("=" * 80)
        print("🎉 Chunks are ready for RAG retrieval!")

    elif next_step == "6":
        print("\n✓ Chunks created but not saved. Exiting...")
    else:
        print("\n✓ Chunks created. Run again to save them.")


if __name__ == "__main__":
    main()
