"""
Patch to disable embeddings by default in all 5 chunking methods
This will prevent quota exceeded errors
"""

import re

# Read the file
with open('text_chunking_methods.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Method 1: Fixed Token Chunking - Add parameter
content = content.replace(
    'def fixed_token_chunking(\n    file_path: str,\n    chunk_size: int = 512,\n    overlap: int = 0,\n    encoding_name: str = "cl100k_base"\n)',
    'def fixed_token_chunking(\n    file_path: str,\n    chunk_size: int = 512,\n    overlap: int = 0,\n    encoding_name: str = "cl100k_base",\n    generate_embeddings: bool = False\n)'
)

# Update fixed_token_chunking embedding generation
content = content.replace(
    '    # Generate embeddings\n    print(f"Generating embeddings for {len(chunks)} chunks...")\n    embeddings = get_gemini_embeddings(chunks)\n\n    print(f"Complete! Created {len(chunks)} chunks with embeddings")',
    '    # Generate embeddings (optional)\n    if generate_embeddings:\n        print(f"Generating embeddings for {len(chunks)} chunks...")\n        embeddings = get_gemini_embeddings(chunks)\n        print(f"Complete! Created {len(chunks)} chunks with embeddings")\n    else:\n        embeddings = []\n        print(f"Complete! Created {len(chunks)} chunks")'
)

# Method 2: Recursive Token Chunking - Add parameter
content = content.replace(
    'def recursive_token_chunking(\n    file_path: str,\n    chunk_size: int = 512,\n    overlap: int = 0,\n    separators: Optional[List[str]] = None,\n    keep_separator: bool = True\n)',
    'def recursive_token_chunking(\n    file_path: str,\n    chunk_size: int = 512,\n    overlap: int = 0,\n    separators: Optional[List[str]] = None,\n    keep_separator: bool = True,\n    generate_embeddings: bool = False\n)'
)

# Update recursive_token_chunking docstring and embedding
old_recursive_embed = '''    # Perform recursive splitting
    chunks = recursive_split(text, separators)

    # Generate embeddings
    print(f"Generating embeddings for {len(chunks)} chunks...")
    embeddings = get_gemini_embeddings(chunks)

    print(f"Complete! Created {len(chunks)} chunks with embeddings")'''

new_recursive_embed = '''    # Perform recursive splitting
    chunks = recursive_split(text, separators)

    # Generate embeddings (optional)
    if generate_embeddings:
        print(f"Generating embeddings for {len(chunks)} chunks...")
        embeddings = get_gemini_embeddings(chunks)
        print(f"Complete! Created {len(chunks)} chunks with embeddings")
    else:
        embeddings = []
        print(f"Complete! Created {len(chunks)} chunks")'''

content = content.replace(old_recursive_embed, new_recursive_embed)

# Method 3: Cluster Semantic Chunking - Add parameter
content = content.replace(
    'def cluster_semantic_chunking(\n    file_path: str,\n    max_chunk_size: int = 400,\n    min_chunk_size: int = 50,\n    overlap: int = 0\n)',
    'def cluster_semantic_chunking(\n    file_path: str,\n    max_chunk_size: int = 400,\n    min_chunk_size: int = 50,\n    overlap: int = 0,\n    generate_embeddings: bool = False\n)'
)

# Method 4: Kamradt Semantic Chunking - Add parameter
content = content.replace(
    'def kamradt_semantic_chunking(\n    file_path: str,\n    avg_chunk_size: int = 400,\n    min_chunk_size: int = 50,\n    overlap: int = 0,\n    buffer_size: int = 3\n)',
    'def kamradt_semantic_chunking(\n    file_path: str,\n    avg_chunk_size: int = 400,\n    min_chunk_size: int = 50,\n    overlap: int = 0,\n    buffer_size: int = 3,\n    generate_embeddings: bool = False\n)'
)

# Write back
with open('text_chunking_methods.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("✓ Added generate_embeddings=False parameter to all 5 methods")
print("✓ Updated embedding generation to be conditional")
print("\nNow manually updating cluster and kamradt embedding logic...")
