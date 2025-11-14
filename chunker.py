#!/usr/bin/env python3
"""
Chunker CLI - Main entry point for document chunking

Usage:
    python chunker.py --file document.md --type recursive
    python chunker.py --dir sample-md-files --output-dir output
    python chunker.py --file document.pdf --type llm --chunk-size 512
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List

from src.chunkers import get_chunker
from src.utils import get_file_text, Config


def get_files_to_process(file_path: str = None, dir_path: str = None) -> List[Path]:
    """Get list of files to process from file or directory"""
    files = []

    if file_path:
        path = Path(file_path)
        if not path.exists():
            print(f"Error: File not found: {file_path}", file=sys.stderr)
            sys.exit(1)
        files.append(path)

    elif dir_path:
        path = Path(dir_path)
        if not path.exists():
            print(f"Error: Directory not found: {dir_path}", file=sys.stderr)
            sys.exit(1)

        # Get all supported file types
        for ext in ['*.md', '*.txt', '*.pdf', '*.docx']:
            files.extend(path.glob(ext))

        if not files:
            print(f"Error: No supported files found in {dir_path}", file=sys.stderr)
            sys.exit(1)

        files = sorted(files)

    return files


def process_file(file_path: Path, args, config, chunker):
    """Process a single file"""
    # Read file
    try:
        text = get_file_text(str(file_path))
        if args.verbose:
            print(f"  ✓ Read {len(text)} characters")
    except Exception as e:
        print(f"  ✗ Error reading file: {e}", file=sys.stderr)
        return None

    # Chunk text
    try:
        metadata = {
            'source_file': str(file_path),
            'chunker_type': args.type,
            'chunk_size': args.chunk_size,
            'chunk_overlap': args.overlap
        }

        chunks = chunker.chunk(text, metadata=metadata)

        if args.verbose:
            total_tokens = sum(c['tokens'] for c in chunks)
            avg_tokens = total_tokens / len(chunks) if chunks else 0
            print(f"  ✓ Created {len(chunks)} chunks (avg {avg_tokens:.1f} tokens)")

        return chunks

    except Exception as e:
        print(f"  ✗ Error chunking: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        return None


def main():
    parser = argparse.ArgumentParser(
        description='Chunk documents using various strategies',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Chunking Types:
  fixed           - Fixed-size token chunks (simple, fast)
  recursive      - Recursive splitting by separators (good default)
  cluster        - Clustering-based semantic chunks (requires embeddings)
  kamradt        - Similarity-based semantic chunks (requires embeddings)
  llm            - LLM-based intelligent semantic chunks (best quality, requires LLM)
  context-aware  - Markdown-aware with context preservation
  section        - Split by markdown sections only

Examples:
  # Single file
  python chunker.py --file document.md

  # Batch process directory
  python chunker.py --dir sample-md-files --output-dir sample-output

  # LLM semantic chunking
  python chunker.py --file document.pdf --type llm --chunk-size 512

  # Custom output directory
  python chunker.py --file document.txt --output-dir my-chunks
        """
    )

    # Input arguments (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--file', '-f',
                            help='Input file path (.txt, .md, .pdf, .docx)')
    input_group.add_argument('--dir', '-d',
                            help='Input directory (processes all supported files)')

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
    parser.add_argument('--output-dir', default='chunks_output',
                       help='Output directory for chunk files (default: chunks_output)')

    parser.add_argument('--no-save', action='store_true',
                       help='Do not save to files, only print summary')

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

    # Get files to process
    files = get_files_to_process(args.file, args.dir)

    print(f"\n{'='*60}")
    print(f"  Chunking Documents")
    print(f"{'='*60}\n")
    print(f"Configuration:")
    print(f"  Chunker type:    {args.type}")
    print(f"  Chunk size:      {args.chunk_size} tokens")
    print(f"  Overlap:         {args.overlap} tokens")
    print(f"  Output dir:      {args.output_dir}")
    print(f"\nProcessing {len(files)} file(s)...\n")

    # Initialize chunker
    try:
        chunker_kwargs = {
            'chunk_size': args.chunk_size,
            'chunk_overlap': args.overlap
        }

        if args.type == 'cluster' and args.num_clusters:
            chunker_kwargs['num_clusters'] = args.num_clusters
        elif args.type == 'kamradt':
            chunker_kwargs['breakpoint_percentile'] = args.breakpoint_percentile

        chunker = get_chunker(args.type, **chunker_kwargs)
        if args.verbose:
            print(f"Initialized: {chunker}\n")
    except Exception as e:
        print(f"Error initializing chunker: {e}", file=sys.stderr)
        sys.exit(1)

    # Create output directory
    if not args.no_save:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)

    # Process files
    stats = {'success': 0, 'failed': 0, 'total_chunks': 0}

    for i, file_path in enumerate(files, 1):
        print(f"[{i}/{len(files)}] {file_path.name}")

        chunks = process_file(file_path, args, config, chunker)

        if chunks is not None:
            stats['success'] += 1
            stats['total_chunks'] += len(chunks)

            # Save to file
            if not args.no_save:
                output_file = output_dir / f"{file_path.stem}_chunks.json"

                output_data = {
                    'source_file': str(file_path),
                    'chunker_type': args.type,
                    'chunk_size': args.chunk_size,
                    'chunk_overlap': args.overlap,
                    'num_chunks': len(chunks),
                    'chunks': chunks
                }

                try:
                    with open(output_file, 'w', encoding='utf-8') as f:
                        json.dump(output_data, f, indent=2, ensure_ascii=False)
                    print(f"  ✓ Saved to {output_file.name}")
                except Exception as e:
                    print(f"  ✗ Error saving: {e}", file=sys.stderr)
                    stats['failed'] += 1
        else:
            stats['failed'] += 1

        print()

    # Summary
    print(f"{'='*60}")
    print(f"  Summary")
    print(f"{'='*60}")
    print(f"  Total files:      {len(files)}")
    print(f"  Successful:       {stats['success']}")
    print(f"  Failed:           {stats['failed']}")
    print(f"  Total chunks:     {stats['total_chunks']}")
    if not args.no_save:
        print(f"  Output directory: {args.output_dir}/")
    print()


if __name__ == '__main__':
    main()
