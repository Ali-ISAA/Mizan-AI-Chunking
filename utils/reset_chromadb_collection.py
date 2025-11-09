"""
Reset ChromaDB Collection
==========================

This script deletes the old collection (with bad page header chunks)
and re-creates it with properly filtered chunks.
"""

import chromadb
from text_chunking_methods import recursive_token_chunking, get_gemini_embeddings
import uuid

def reset_collection():
    print("=" * 80)
    print("RESETTING CHROMADB COLLECTION")
    print("=" * 80)

    # Initialize client
    client = chromadb.CloudClient(
        api_key='ck-8PeJWvgo9YpuTWwwKPMNtCeKc77MkuZq2N8R31ryjHrh',
        tenant='cc20128c-e79a-47f4-a0cd-ccd549e2f6a9',
        database='DEV'
    )

    collection_name = "my_gemini_collection"

    # Step 1: Delete old collection
    print(f"\nStep 1: Deleting old collection '{collection_name}'...")
    try:
        client.delete_collection(name=collection_name)
        print(f"✓ Deleted old collection")
    except Exception as e:
        print(f"Note: Collection may not exist: {e}")

    # Step 2: Generate fresh chunks with embeddings
    print(f"\nStep 2: Generating fresh chunks from PDF...")
    chunks, embeddings = recursive_token_chunking(
        file_path="Digital Government Policies - V2.0.pdf",
        chunk_size=512,
        overlap=50,
        generate_embeddings=True
    )
    print(f"✓ Generated {len(chunks)} chunks with embeddings")

    # Step 3: Filter out page headers and short chunks
    print(f"\nStep 3: Filtering out page headers and short chunks...")
    min_chunk_length = 50
    filtered_chunks = []
    filtered_embeddings = []

    for chunk, embedding in zip(chunks, embeddings):
        # Remove page headers and very short chunks
        if len(chunk.strip()) >= min_chunk_length and not chunk.strip().startswith("--- Page"):
            filtered_chunks.append(chunk)
            filtered_embeddings.append(embedding)

    print(f"Original chunks: {len(chunks)}")
    print(f"Filtered chunks: {len(filtered_chunks)}")
    print(f"Removed: {len(chunks) - len(filtered_chunks)} short/header chunks")

    # Step 4: Create new collection and add filtered chunks
    print(f"\nStep 4: Creating new collection and storing chunks...")
    collection = client.get_or_create_collection(name=collection_name)

    # Generate IDs
    ids = [str(uuid.uuid4()) for _ in filtered_embeddings]

    # Add to ChromaDB
    collection.add(
        ids=ids,
        documents=filtered_chunks,
        embeddings=filtered_embeddings
    )

    print(f"✓ Stored {len(filtered_chunks)} chunks in ChromaDB")

    # Step 5: Verify with a test query
    print(f"\nStep 5: Testing with sample query...")
    test_query = "digital transformation"
    query_embedding = get_gemini_embeddings([test_query])[0]

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=1
    )

    print(f"\nTest Query: '{test_query}'")
    print(f"Top Result Length: {len(results['documents'][0][0])} characters")
    print(f"Top Result Preview: {results['documents'][0][0][:150]}...")

    print("\n" + "=" * 80)
    print("COLLECTION RESET COMPLETE!")
    print("=" * 80)
    print(f"\nYou can now query the collection with:")
    print(f"  python test_chromadb_query.py")

if __name__ == "__main__":
    reset_collection()
