#!/usr/bin/env python3
"""
Chunker CLI - Main entry point for document chunking

Usage:
    python chunker.py --file document.md --type recursive
    python chunker.py --file document.pdf --type llm --chunk-size 512
    python chunker.py --file document.txt --type fixed --output chunks.json
"""

import argparse
import json
import sys
from pathlib import Path

from src.chunkers import get_chunker
from src.utils import get_file_text, Config


def main():
    parser = argparse.ArgumentParser(
        description='Chunk documents using various strategies',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Chunking Types:
  fixed          - Fixed-size token chunks (simple, fast)
  recursive      - Recursive splitting by separators (good default)
  cluster        - Clustering-based semantic chunks (requires embeddings)
  kamradt        - Similarity-based semantic chunks (requires embeddings)
  llm            - LLM-based intelligent semantic chunks (best quality, requires LLM)
  context-aware  - Markdown-aware with context preservation
  section        - Split by markdown sections only

Examples:
  # Basic recursive chunking (recommended default)
  python chunker.py --file document.md

  # LLM semantic chunking with custom size
  python chunker.py --file document.pdf --type llm --chunk-size 512

  # Fixed chunking with overlap and output to file
  python chunker.py --file document.txt --type fixed --chunk-size 256 --overlap 50 --output chunks.json

  # Cluster semantic chunking
  python chunker.py --file document.md --type cluster --num-clusters 20
        """
    )

    # Required arguments
    parser.add_argument('--file', '-f', required=True,
                       help='Input file path (.txt, .md, .pdf, .docx)')

    # Chunking configuration
    parser.add_argument('--type', '-t', default='recursive',
                       choices=['fixed', 'recursive', 'cluster', 'kamradt', 'llm', 'context-aware', 'section'],
                       help='Chunking type (default: recursive)')

    parser.add_argument('--chunk-size', '-s', type=int, default=512,
                       help='Target chunk size in tokens (default: 512)')

    parser.add_argument('--overlap', '-o', type=int, default=50,
                       help='Chunk overlap in tokens (default: 50)')

    # Type-specific arguments
    parser.add_argument('--num-clusters', type=int,
                       help='Number of clusters for cluster chunking (auto if not specified)')

    parser.add_argument('--breakpoint-percentile', type=int, default=95,
                       help='Breakpoint percentile for Kamradt chunking (default: 95)')

    # Output configuration
    parser.add_argument('--output', '-out',
                       help='Output file path (JSON format). If not specified, prints to stdout')

    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose output')

    # Configuration
    parser.add_argument('--env-file',
                       help='Path to .env file (default: .env in project root)')

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

    # Read file
    if not Path(args.file).exists():
        print(f"Error: File not found: {args.file}", file=sys.stderr)
        sys.exit(1)

    print(f"Reading file: {args.file}")
    try:
        text = get_file_text(args.file)
        print(f"✓ Read {len(text)} characters")
    except Exception as e:
        print(f"Error reading file: {e}", file=sys.stderr)
        sys.exit(1)

    # Initialize chunker
    print(f"\nInitializing {args.type} chunker...")
    try:
        chunker_kwargs = {
            'chunk_size': args.chunk_size,
            'chunk_overlap': args.overlap
        }

        # Add type-specific arguments
        if args.type == 'cluster' and args.num_clusters:
            chunker_kwargs['num_clusters'] = args.num_clusters
        elif args.type == 'kamradt':
            chunker_kwargs['breakpoint_percentile'] = args.breakpoint_percentile

        chunker = get_chunker(args.type, **chunker_kwargs)
        print(f"✓ {chunker}")
    except Exception as e:
        print(f"Error initializing chunker: {e}", file=sys.stderr)
        sys.exit(1)

    # Chunk text
    print(f"\nChunking text...")
    try:
        metadata = {
            'source_file': args.file,
            'chunker_type': args.type,
            'chunk_size': args.chunk_size,
            'chunk_overlap': args.overlap
        }

        chunks = chunker.chunk(text, metadata=metadata)
        print(f"\n✓ Created {len(chunks)} chunks")

        # Statistics
        if args.verbose:
            total_tokens = sum(c['tokens'] for c in chunks)
            avg_tokens = total_tokens / len(chunks) if chunks else 0
            min_tokens = min(c['tokens'] for c in chunks) if chunks else 0
            max_tokens = max(c['tokens'] for c in chunks) if chunks else 0

            print(f"\nChunk Statistics:")
            print(f"  Total chunks:  {len(chunks)}")
            print(f"  Total tokens:  {total_tokens}")
            print(f"  Average size:  {avg_tokens:.1f} tokens")
            print(f"  Min size:      {min_tokens} tokens")
            print(f"  Max size:      {max_tokens} tokens")

    except Exception as e:
        print(f"Error chunking text: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

    # Output
    output_data = {
        'file': args.file,
        'chunker_type': args.type,
        'chunk_size': args.chunk_size,
        'chunk_overlap': args.overlap,
        'num_chunks': len(chunks),
        'chunks': chunks
    }

    if args.output:
        print(f"\nSaving to: {args.output}")
        try:
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
            print(f"✓ Saved successfully")
        except Exception as e:
            print(f"Error saving output: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        # Print to stdout
        print("\n" + "="*80)
        print(json.dumps(output_data, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
