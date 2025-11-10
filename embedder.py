#!/usr/bin/env python3
"""
Embedder CLI - Main entry point for embedding and storing documents

Usage:
    python embedder.py --file document.md
    python embedder.py --chunks chunks.json --vector-store chromadb
    python embedder.py --file document.pdf --chunker-type llm --vector-store supabase
"""

import argparse
import json
import sys
from pathlib import Path

from src.chunkers import get_chunker
from src.embedders import get_embedder
from src.vector_stores import get_vector_store
from src.utils import get_file_text, Config


def main():
    parser = argparse.ArgumentParser(
        description='Embed and store document chunks in vector database',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Vector Stores:
  chromadb       - ChromaDB (local or cloud)
  supabase       - Supabase with pgvector
  pgvector       - PostgreSQL with pgvector
  qdrant         - Qdrant vector database
  weaviate       - Weaviate vector database
  pinecone       - Pinecone vector database

Examples:
  # Embed document and store in ChromaDB (uses settings from .env)
  python embedder.py --file document.md

  # Use specific chunker and vector store
  python embedder.py --file document.pdf --chunker-type llm --vector-store supabase

  # Load pre-chunked data
  python embedder.py --chunks chunks.json --vector-store qdrant

  # Custom collection name
  python embedder.py --file document.txt --collection my_docs
        """
    )

    # Input source (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--file', '-f',
                            help='Input file path (.txt, .md, .pdf, .docx)')
    input_group.add_argument('--chunks', '-c',
                            help='Pre-chunked JSON file (output from chunker.py)')

    # Chunking configuration (only used if --file is provided)
    parser.add_argument('--chunker-type', '-t', default='recursive',
                       choices=['fixed', 'recursive', 'cluster', 'kamradt', 'llm', 'context-aware', 'section'],
                       help='Chunking type (default: recursive, only used with --file)')

    parser.add_argument('--chunk-size', '-s', type=int, default=512,
                       help='Chunk size in tokens (default: 512)')

    parser.add_argument('--chunk-overlap', '-o', type=int, default=50,
                       help='Chunk overlap in tokens (default: 50)')

    # Vector store configuration
    parser.add_argument('--vector-store', '-vs',
                       choices=['chromadb', 'supabase', 'pgvector', 'qdrant', 'weaviate', 'pinecone'],
                       help='Vector store type (default: from .env VECTOR_STORE)')

    parser.add_argument('--collection', '--table', '--index',
                       help='Collection/table/index name (default: from .env COLLECTION_NAME or filename)')

    # Embedding configuration
    parser.add_argument('--embedding-provider',
                       choices=['gemini', 'openai', 'ollama'],
                       help='Embedding provider (default: from .env EMBEDDING_PROVIDER)')

    parser.add_argument('--embedding-model',
                       help='Embedding model name (default: from .env EMBEDDING_MODEL)')

    # Other options
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose output')

    parser.add_argument('--env-file',
                       help='Path to .env file (default: .env in project root)')

    parser.add_argument('--skip-existing', action='store_true',
                       help='Skip if collection already exists')

    args = parser.parse_args()

    # Initialize config
    try:
        if args.env_file:
            config = Config(args.env_file)
        else:
            config = Config()
    except Exception as e:
        print(f"Error loading configuration: {e}", file=sys.stderr)
        print("Please ensure .env file is properly configured.", file=sys.stderr)
        sys.exit(1)

    # Get configuration values
    vector_store_type = args.vector_store or config.vector_store
    embedding_provider = args.embedding_provider or config.embedding_provider
    embedding_model = args.embedding_model or config.embedding_model
    embedding_dimension = config.embedding_dimension

    print(f"Configuration:")
    print(f"  Vector Store:  {vector_store_type}")
    print(f"  Embedder:      {embedding_provider}/{embedding_model}")
    print(f"  Dimension:     {embedding_dimension}")

    # Load or create chunks
    if args.chunks:
        # Load pre-chunked data
        print(f"\nLoading chunks from: {args.chunks}")
        try:
            with open(args.chunks, 'r', encoding='utf-8') as f:
                data = json.load(f)
                chunks = data['chunks']
                source_file = data.get('file', args.chunks)
            print(f"✓ Loaded {len(chunks)} chunks")
        except Exception as e:
            print(f"Error loading chunks: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        # Read and chunk file
        if not Path(args.file).exists():
            print(f"Error: File not found: {args.file}", file=sys.stderr)
            sys.exit(1)

        print(f"\nReading file: {args.file}")
        try:
            text = get_file_text(args.file)
            print(f"✓ Read {len(text)} characters")
        except Exception as e:
            print(f"Error reading file: {e}", file=sys.stderr)
            sys.exit(1)

        # Chunk text
        print(f"\nChunking with {args.chunker_type} chunker...")
        try:
            chunker = get_chunker(
                args.chunker_type,
                chunk_size=args.chunk_size,
                chunk_overlap=args.chunk_overlap
            )
            chunks = chunker.chunk(text, metadata={'source_file': args.file})
            print(f"✓ Created {len(chunks)} chunks")
        except Exception as e:
            print(f"Error chunking: {e}", file=sys.stderr)
            sys.exit(1)

        source_file = args.file

    # Determine collection name
    if args.collection:
        collection_name = args.collection
    else:
        # Auto-generate from filename
        collection_name = Path(source_file).stem.replace('-', '_').replace('.', '_')

    print(f"  Collection:    {collection_name}")

    # Initialize embedder
    print(f"\nInitializing embedder...")
    try:
        embedder = get_embedder(
            embedding_provider,
            embedding_model,
            embedding_dimension
        )
        print(f"✓ {embedder}")
    except Exception as e:
        print(f"Error initializing embedder: {e}", file=sys.stderr)
        sys.exit(1)

    # Initialize vector store
    print(f"\nInitializing vector store...")
    try:
        vector_store = get_vector_store(
            vector_store_type,
            collection_name,
            embedding_dimension
        )
        print(f"✓ {vector_store}")

        # Create collection
        if args.skip_existing:
            try:
                count = vector_store.get_count()
                print(f"  Collection already exists with {count} vectors, skipping...")
                return
            except:
                pass  # Collection doesn't exist, continue

        created = vector_store.create_collection()
        if created:
            print(f"✓ Created new collection")
        else:
            print(f"✓ Collection already exists")

    except Exception as e:
        print(f"Error initializing vector store: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

    # Generate embeddings
    print(f"\nGenerating embeddings for {len(chunks)} chunks...")
    try:
        texts = [chunk['text'] for chunk in chunks]
        embeddings = embedder.embed_batch(texts)
        print(f"✓ Generated {len(embeddings)} embeddings")
    except Exception as e:
        print(f"Error generating embeddings: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

    # Store in vector database
    print(f"\nStoring in {vector_store_type}...")
    try:
        metadata_list = [chunk['metadata'] for chunk in chunks]

        success = vector_store.insert(
            texts=texts,
            embeddings=embeddings,
            metadata=metadata_list
        )

        if success:
            print(f"✓ Successfully stored {len(chunks)} chunks")
            final_count = vector_store.get_count()
            print(f"✓ Total vectors in collection: {final_count}")
        else:
            print(f"Error: Failed to store chunks", file=sys.stderr)
            sys.exit(1)

    except Exception as e:
        print(f"Error storing chunks: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

    print(f"\n✅ Complete! Collection '{collection_name}' ready for search.")


if __name__ == '__main__':
    main()
