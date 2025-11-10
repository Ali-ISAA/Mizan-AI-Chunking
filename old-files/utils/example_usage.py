"""
Example Usage of Text Chunking Methods
=======================================

This script demonstrates how to use all 5 chunking methods
with the provided PDF file.

Google API Key is pre-configured in text_chunking_methods.py
"""
import chromadb
import uuid
import os
from text_chunking_methods import (
    fixed_token_chunking,
    recursive_token_chunking,
    cluster_semantic_chunking,
    kamradt_semantic_chunking,
    llm_semantic_chunking,
    get_gemini_embeddings
)

# Path to the test PDF file
PDF_FILE = "Digital Government Policies - V2.0.pdf"


def filter_chunks(chunks, embeddings, min_length=50):
    """
    Filter out empty chunks, page headers, and very short chunks.

    Parameters:
    -----------
    chunks : List[str]
        List of text chunks
    embeddings : List[List[float]]
        List of embeddings corresponding to chunks
    min_length : int
        Minimum character length for a valid chunk (default: 50)

    Returns:
    --------
    Tuple[List[str], List[List[float]]]
        Filtered chunks and embeddings
    """
    filtered_chunks = []
    filtered_embeddings = []

    for chunk, embedding in zip(chunks, embeddings):
        chunk_stripped = chunk.strip()
        # Filter out: page headers, very short chunks, empty chunks
        if (len(chunk_stripped) >= min_length and
            not chunk_stripped.startswith("--- Page") and
            chunk_stripped != ""):
            filtered_chunks.append(chunk)
            filtered_embeddings.append(embedding)

    print(f"   Original chunks: {len(chunks)}")
    print(f"   Filtered chunks: {len(filtered_chunks)}")
    print(f"   Removed: {len(chunks) - len(filtered_chunks)} short/header chunks")

    return filtered_chunks, filtered_embeddings


def save_to_chromadb(chunks, embeddings, collection_name):
    """
    Save filtered chunks and embeddings to ChromaDB.

    Parameters:
    -----------
    chunks : List[str]
        List of text chunks
    embeddings : List[List[float]]
        List of embeddings
    collection_name : str
        Name of the ChromaDB collection

    Returns:
    --------
    int
        Number of chunks saved
    """
    # Initialize Chroma Cloud client
    client = chromadb.CloudClient(
        api_key='ck-8PeJWvgo9YpuTWwwKPMNtCeKc77MkuZq2N8R31ryjHrh',
        tenant='cc20128c-e79a-47f4-a0cd-ccd549e2f6a9',
        database='DEV'
    )

    # Create or get collection
    collection = client.get_or_create_collection(name=collection_name)

    # Filter out empty or very short chunks
    print(f"\n   Filtering chunks (minimum length: 50 characters)...")
    filtered_chunks, filtered_embeddings = filter_chunks(chunks, embeddings, min_length=50)

    # Generate unique string IDs
    ids = [str(uuid.uuid4()) for _ in filtered_embeddings]

    # Add embeddings to ChromaDB
    collection.add(ids=ids, documents=filtered_chunks, embeddings=filtered_embeddings)
    print(f"   ✅ {len(filtered_chunks)} embeddings saved to ChromaDB collection '{collection_name}'")

    return len(filtered_chunks)


def query_chromadb(query_text: str, collection_name: str = "my_gemini_collection", n_results: int = 3):
    """
    Query ChromaDB collection using Gemini embeddings.

    Parameters:
    -----------
    query_text : str
        The search query text
    collection_name : str
        Name of the ChromaDB collection to query (default: "my_gemini_collection")
    n_results : int
        Number of similar results to return (default: 3)

    Returns:
    --------
    dict
        Query results containing documents, ids, and distances

    Example:
    --------
    >>> results = query_chromadb("digital government policies", n_results=5)
    >>> print(results['documents'])
    """
    print("\n" + "=" * 80)
    print("QUERYING CHROMADB")
    print("=" * 80)
    print(f"Query: '{query_text}'")
    print(f"Collection: {collection_name}")
    print(f"Top results: {n_results}\n")

    # Initialize Chroma Cloud client
    client = chromadb.CloudClient(
        api_key='ck-8PeJWvgo9YpuTWwwKPMNtCeKc77MkuZq2N8R31ryjHrh',
        tenant='cc20128c-e79a-47f4-a0cd-ccd549e2f6a9',
        database='DEV'
    )

    # Get the collection
    collection = client.get_collection(name=collection_name)

    # Embed the query using the same Gemini model as your chunks
    print("Embedding query using Gemini...")
    query_embedding = get_gemini_embeddings([query_text])[0]  # Returns a 768-dim vector
    print(f"Query embedding dimension: {len(query_embedding)}")

    # Perform query on your collection
    print(f"Searching for top {n_results} similar chunks...\n")
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results
    )

    # Display results
    print("=" * 80)
    print("SEARCH RESULTS")
    print("=" * 80)

    for i, (doc, doc_id, distance) in enumerate(zip(
        results['documents'][0],
        results['ids'][0],
        results['distances'][0]
    )):
        print(f"\n[Result {i+1}]")
        print(f"ID: {doc_id}")
        print(f"Distance: {distance:.4f}")
        print(f"Document Length: {len(doc)} characters")
        print("\n--- COMPLETE DOCUMENT ---")
        print(doc)  # Print complete document
        print("--- END OF DOCUMENT ---")
        print("-" * 80)

    return results


def main():
    print("=" * 80)
    print("TEXT CHUNKING METHODS - EXAMPLE USAGE")
    print("=" * 80)
    print(f"\nTest File: {PDF_FILE}")

    if not os.path.exists(PDF_FILE):
        print(f"\n❌ Error: File '{PDF_FILE}' not found!")
        print("Please ensure the PDF file is in the same directory.")
        return

    print(f"✅ File found!\n")

    # ========================================================================
    # Example 1: Fixed Token Chunking
    # ========================================================================
    print("\n" + "=" * 80)
    print("1. FIXED TOKEN CHUNKING")
    print("=" * 80)
    print("Description: Splits text into fixed-size token chunks")
    print("Best for: High-volume processing, strict token limits\n")

    try:
        chunks, embeddings = fixed_token_chunking(
            file_path=PDF_FILE,
            chunk_size=512,
            overlap=50,
            generate_embeddings=True  # Generate embeddings for ChromaDB
        )

        print(f"\n📊 Results:")
        print(f"   Total chunks: {len(chunks)}")
        if embeddings and len(embeddings) > 0:
            print(f"   Embedding dimension: {len(embeddings[0])}")
        else:
            print(f"   Embeddings: Not generated")
        print(f"\n   First chunk preview (200 chars):")
        print(f"   {chunks[0][:200]}...")

        # Save chunks to file
        output_file = "output_fixed_token_chunks.txt"
        with open(output_file, 'w', encoding='utf-8') as f:
            for i, chunk in enumerate(chunks, 1):
                f.write(f"\n{'='*60}\n")
                f.write(f"CHUNK {i}\n")
                f.write(f"{'='*60}\n")
                f.write(chunk)
                f.write(f"\n\n")

        print(f"\n   ✅ Chunks saved to: {output_file}")

        # Save to ChromaDB
        if embeddings and len(embeddings) > 0:
            save_to_chromadb(chunks, embeddings, "fixed_token_collection")

    except Exception as e:
        print(f"\n   ❌ Error: {e}")

    # ========================================================================
    # Example 2: Recursive Token Chunking
    # ========================================================================
    print("\n\n" + "=" * 80)
    print("2. RECURSIVE TOKEN CHUNKING")
    print("=" * 80)
    print("Description: Splits at natural boundaries (paragraphs, sentences)")
    print("Best for: Maintaining document structure\n")

    try:
        chunks, embeddings = recursive_token_chunking(
            file_path=PDF_FILE,
            chunk_size=400,
            overlap=200,
            separators=["\n\n", "\n", ". ", "? ", "! ", " ", ""],
            generate_embeddings=True  # Set to True if you need embeddings
        )

        print(f"\n📊 Results:")
        print(f"   Total chunks: {len(chunks)}")
        if embeddings and len(embeddings) > 0:
            print(f"   Embedding dimension: {len(embeddings[0])}")
        else:
            print(f"   Embeddings: Not generated (generate_embeddings=False)")
        print(f"\n   First 3 chunks preview:")
        for i, chunk in enumerate(chunks[:3], 1):
            print(f"\n   Chunk {i} ({len(chunk)} chars):")
            print(f"   {chunk[:150]}...")

        # Save chunks to file
        output_file = "output_recursive_chunks.txt"
        with open(output_file, 'w', encoding='utf-8') as f:
            for i, chunk in enumerate(chunks, 1):
                f.write(f"\n{'='*60}\n")
                f.write(f"CHUNK {i}\n")
                f.write(f"{'='*60}\n")
                f.write(chunk)
                f.write(f"\n\n")

        print(f"\n   ✅ Chunks saved to: {output_file}")

        # Save to ChromaDB
        if embeddings and len(embeddings) > 0:
            save_to_chromadb(chunks, embeddings, "recursive_token_collection")

        # # Example query text you want to search for
        # query_text = "Your search query here"

        # # Optionally, you can embed your query first using the same embedding model you used for your data
        # # If you have an embedding function or precomputed query embedding, you can pass embeddings instead of text
        
        # # Embed the query using the same Gemini model as your chunks
        # query_embeddings = get_gemini_embeddings([query_text])[0]  # Returns a 768-dim vector

        # # Perform query on your collection for the top n results (e.g., 3)
        # #query_texts=[query_text] if query_embeddings is None else None,
            
        # results = collection.query(
        #     query_embeddings=[query_embeddings] if query_embeddings is not None else None,
        #     n_results=3  # number of similar chunks to return
        # )

        # # The results dictionary contains matching ids, documents, embeddings, and scores
        # print("Search results (documents):", results['documents'])
        # print("Corresponding IDs:", results['ids'])
        # print("Similarity scores:", results['distances'])










        

    except Exception as e:
        print(f"\n   ❌ Error: {e}")

    # ========================================================================
    # Example 3: Cluster Semantic Chunking
    # ========================================================================
    print("\n\n" + "=" * 80)
    print("3. CLUSTER SEMANTIC CHUNKING")
    print("=" * 80)
    print("Description: Groups semantically similar content using embeddings")
    print("Best for: RAG systems, knowledge bases\n")

    try:
        chunks, embeddings = cluster_semantic_chunking(
            file_path=PDF_FILE,
            max_chunk_size=400,
            overlap=50,
            generate_embeddings=True  # Generate embeddings for ChromaDB
        )

        print(f"\n📊 Results:")
        print(f"   Total chunks: {len(chunks)}")
        if embeddings and len(embeddings) > 0:
            print(f"   Embedding dimension: {len(embeddings[0])}")
        else:
            print(f"   Embeddings: Not generated")

        # Save chunks
        output_file = "output_cluster_semantic_chunks.txt"
        with open(output_file, 'w', encoding='utf-8') as f:
            for i, chunk in enumerate(chunks, 1):
                f.write(f"\n{'='*60}\n")
                f.write(f"CHUNK {i}\n")
                f.write(f"{'='*60}\n")
                f.write(chunk)
                f.write(f"\n\n")

        print(f"\n   ✅ Chunks saved to: {output_file}")

        # Save to ChromaDB
        if embeddings and len(embeddings) > 0:
            save_to_chromadb(chunks, embeddings, "cluster_semantic_collection")

    except Exception as e:
        print(f"\n   ❌ Error: {e}")

    # ========================================================================
    # Example 4: Kamradt Semantic Chunking
    # ========================================================================
    print("\n\n" + "=" * 80)
    print("4. KAMRADT SEMANTIC CHUNKING")
    print("=" * 80)
    print("Description: Splits at topic boundaries using similarity")
    print("Best for: Documents with clear topic shifts\n")

    try:
        chunks, embeddings = kamradt_semantic_chunking(
            file_path=PDF_FILE,
            avg_chunk_size=300,
            overlap=30,
            generate_embeddings=True  # Generate embeddings for ChromaDB
        )

        print(f"\n📊 Results:")
        print(f"   Total chunks: {len(chunks)}")
        if embeddings and len(embeddings) > 0:
            print(f"   Embedding dimension: {len(embeddings[0])}")
        else:
            print(f"   Embeddings: Not generated")

        # Save chunks
        output_file = "output_kamradt_chunks.txt"
        with open(output_file, 'w', encoding='utf-8') as f:
            for i, chunk in enumerate(chunks, 1):
                f.write(f"\n{'='*60}\n")
                f.write(f"CHUNK {i}\n")
                f.write(f"{'='*60}\n")
                f.write(chunk)
                f.write(f"\n\n")

        print(f"\n   ✅ Chunks saved to: {output_file}")

        # Save to ChromaDB
        if embeddings and len(embeddings) > 0:
            save_to_chromadb(chunks, embeddings, "kamradt_semantic_collection")

    except Exception as e:
        print(f"\n   ❌ Error: {e}")

    # ========================================================================
    # Example 5: LLM Semantic Chunking
    # ========================================================================
    print("\n\n" + "=" * 80)
    print("5. LLM SEMANTIC CHUNKING (Takes longer - uses Gemini LLM)")
    print("=" * 80)
    print("Description: Uses Gemini LLM for intelligent splitting")
    print("Best for: Premium applications requiring best quality")
    print("\n⚠️  Note: This method is slower but produces highest quality chunks!\n")

    try:
        chunks, embeddings = llm_semantic_chunking(
            file_path=PDF_FILE,
            overlap=50,
            model_name="gemini-2.5-flash",  # Fast stable model
            generate_embeddings=True  # Generate embeddings for ChromaDB
        )

        print(f"\n📊 Results:")
        print(f"   Total chunks: {len(chunks)}")
        if embeddings and len(embeddings) > 0:
            print(f"   Embedding dimension: {len(embeddings[0])}")
        else:
            print(f"   Embeddings: Not generated")

        # Save chunks
        output_file = "output_llm_semantic_chunks.txt"
        with open(output_file, 'w', encoding='utf-8') as f:
            for i, chunk in enumerate(chunks, 1):
                f.write(f"\n{'='*60}\n")
                f.write(f"CHUNK {i}\n")
                f.write(f"{'='*60}\n")
                f.write(chunk)
                f.write(f"\n\n")

        print(f"\n   ✅ Chunks saved to: {output_file}")

        # Save to ChromaDB
        if embeddings and len(embeddings) > 0:
            save_to_chromadb(chunks, embeddings, "llm_semantic_collection")

    except Exception as e:
        print(f"\n   ❌ Error: {e}")

    # ========================================================================
    # Summary
    # ========================================================================
    print("\n\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("\n✅ Successfully demonstrated ALL 5 chunking methods!")
    print("\nFiles created:")
    print("   - output_fixed_token_chunks.txt")
    print("   - output_recursive_chunks.txt")
    print("   - output_cluster_semantic_chunks.txt")
    print("   - output_kamradt_chunks.txt")
    print("   - output_llm_semantic_chunks.txt")
    print("\n📝 Next Steps:")
    print("   1. Review the output files to compare chunking results")
    print("   2. Experiment with different chunk_size and overlap values")
    print("   3. Use the best method for your specific use case")
    print("   4. Integrate into your RAG or embedding pipeline")
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
