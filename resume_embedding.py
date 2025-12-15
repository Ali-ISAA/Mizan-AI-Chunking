#!/usr/bin/env python3
"""
Resume embedding for files that failed during Qdrant crash.
Uses mxbai-embed-large (1024 dims) into alibaba_docs collection.
"""

import json
import requests
import time
from pathlib import Path

OLLAMA_URL = 'http://localhost:11434/api/embeddings'
MODEL = 'mxbai-embed-large'
COLLECTION = 'alibaba_docs'
DIMENSION = 1024

def embed_text(text):
    response = requests.post(OLLAMA_URL, json={
        'model': MODEL,
        'prompt': text
    }, timeout=120)
    response.raise_for_status()
    return response.json()['embedding']

def main():
    from src.vector_stores import get_vector_store

    # Load files to resume
    with open('resume_files.txt', 'r') as f:
        files_to_embed = [line.strip() for line in f if line.strip()]

    print(f"Resuming embedding for {len(files_to_embed)} files")
    print(f"Model: {MODEL} ({DIMENSION} dims)")
    print(f"Collection: {COLLECTION}")
    print()

    # Initialize vector store
    vector_store = get_vector_store('qdrant', COLLECTION, DIMENSION)

    success = 0
    failed = 0
    failed_files = []

    for i, chunk_file in enumerate(files_to_embed, 1):
        name = Path(chunk_file).name[:55]
        print(f'[{i}/{len(files_to_embed)}] {name}...', end=' ', flush=True)

        try:
            # Load chunks
            with open(chunk_file, 'r') as f:
                data = json.load(f)
            chunks = data.get('chunks', [])

            if not chunks:
                print('skip (no chunks)')
                continue

            # Embed
            texts = [c['text'] for c in chunks]
            embeddings = []
            for text in texts:
                emb = embed_text(text)
                embeddings.append(emb)

            # Store
            metadata_list = [c['metadata'] for c in chunks]
            result = vector_store.insert(
                texts=texts,
                embeddings=embeddings,
                metadata=metadata_list
            )

            if result:
                print(f'✓ {len(chunks)} chunks')
                success += 1
            else:
                print('✗ storage failed')
                failed += 1
                failed_files.append(chunk_file)

        except Exception as e:
            print(f'✗ {str(e)[:50]}')
            failed += 1
            failed_files.append(chunk_file)

        time.sleep(0.1)

    print(f'\n{"="*60}')
    print(f'Success: {success}/{len(files_to_embed)}')
    print(f'Failed:  {failed}/{len(files_to_embed)}')

    if failed_files:
        with open('still_failed.txt', 'w') as f:
            for path in failed_files:
                f.write(path + '\n')
        print(f'Failed files saved to still_failed.txt')

if __name__ == '__main__':
    main()
