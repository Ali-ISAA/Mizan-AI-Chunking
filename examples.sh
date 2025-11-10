#!/bin/bash
# =============================================================================
# MizanAI Chunking - Usage Examples
# =============================================================================
# This file contains various usage examples for the chunker and embedder

echo "================================"
echo "MizanAI Chunking - Usage Examples"
echo "================================"
echo ""

# =============================================================================
# CHUNKING EXAMPLES
# =============================================================================

echo "--- CHUNKING EXAMPLES ---"
echo ""

# 1. Basic recursive chunking (recommended default)
echo "1. Basic recursive chunking:"
echo "python chunker.py --file document.md"
echo ""

# 2. Fixed token chunking with overlap
echo "2. Fixed token chunking with overlap:"
echo "python chunker.py --file document.txt --type fixed --chunk-size 256 --overlap 50"
echo ""

# 3. LLM semantic chunking
echo "3. LLM semantic chunking (best quality):"
echo "python chunker.py --file document.pdf --type llm --chunk-size 512"
echo ""

# 4. Context-aware markdown chunking
echo "4. Context-aware markdown chunking:"
echo "python chunker.py --file document.md --type context-aware"
echo ""

# 5. Section-based chunking
echo "5. Section-based chunking:"
echo "python chunker.py --file document.md --type section"
echo ""

# 6. Cluster semantic chunking
echo "6. Cluster semantic chunking:"
echo "python chunker.py --file document.txt --type cluster --num-clusters 20"
echo ""

# 7. Kamradt semantic chunking
echo "7. Kamradt semantic chunking:"
echo "python chunker.py --file document.md --type kamradt --breakpoint-percentile 95"
echo ""

# 8. Save chunks to JSON file
echo "8. Save chunks to JSON file:"
echo "python chunker.py --file document.txt --type recursive --output chunks.json"
echo ""

# =============================================================================
# EMBEDDING & STORAGE EXAMPLES
# =============================================================================

echo "--- EMBEDDING & STORAGE EXAMPLES ---"
echo ""

# 9. Basic embedding and storage (uses .env settings)
echo "9. Basic embedding and storage:"
echo "python embedder.py --file document.md"
echo ""

# 10. Specify chunker and vector store
echo "10. Specify chunker and vector store:"
echo "python embedder.py --file document.pdf --chunker-type llm --vector-store supabase"
echo ""

# 11. Use ChromaDB
echo "11. Store in ChromaDB:"
echo "python embedder.py --file document.txt --vector-store chromadb --collection my_docs"
echo ""

# 12. Use Supabase
echo "12. Store in Supabase:"
echo "python embedder.py --file document.md --vector-store supabase"
echo ""

# 13. Use Qdrant
echo "13. Store in Qdrant:"
echo "python embedder.py --file document.pdf --vector-store qdrant"
echo ""

# 14. Use Pinecone
echo "14. Store in Pinecone:"
echo "python embedder.py --file document.txt --vector-store pinecone --collection docs_index"
echo ""

# 15. Load pre-chunked data
echo "15. Load and embed pre-chunked data:"
echo "python embedder.py --chunks chunks.json --vector-store chromadb"
echo ""

# 16. Custom embedding provider
echo "16. Use OpenAI embeddings:"
echo "python embedder.py --file document.md --embedding-provider openai --embedding-model text-embedding-3-small"
echo ""

# 17. Use Ollama (local) embeddings
echo "17. Use Ollama local embeddings:"
echo "python embedder.py --file document.txt --embedding-provider ollama --embedding-model nomic-embed-text"
echo ""

# =============================================================================
# COMPLETE WORKFLOWS
# =============================================================================

echo "--- COMPLETE WORKFLOWS ---"
echo ""

# 18. Workflow 1: Chunk, then embed later
echo "18. Two-step workflow (chunk first, embed later):"
echo "python chunker.py --file document.md --type llm --output chunks.json"
echo "python embedder.py --chunks chunks.json --vector-store chromadb"
echo ""

# 19. Workflow 2: All-in-one with ChromaDB
echo "19. All-in-one workflow with ChromaDB:"
echo "python embedder.py --file document.pdf --chunker-type recursive --vector-store chromadb"
echo ""

# 20. Workflow 3: All-in-one with Supabase
echo "20. All-in-one workflow with Supabase:"
echo "python embedder.py --file document.md --chunker-type llm --vector-store supabase"
echo ""

# 21. Workflow 4: Process multiple files
echo "21. Process multiple files:"
echo "for file in docs/*.md; do"
echo "  python embedder.py --file \"\$file\" --vector-store chromadb"
echo "done"
echo ""

# =============================================================================
# ADVANCED EXAMPLES
# =============================================================================

echo "--- ADVANCED EXAMPLES ---"
echo ""

# 22. Custom chunk sizes for different content types
echo "22. Large chunks for code documentation:"
echo "python embedder.py --file api_docs.md --chunk-size 1024 --chunk-overlap 100"
echo ""

# 23. Small chunks for Q&A
echo "23. Small chunks for Q&A pairs:"
echo "python embedder.py --file faq.txt --chunk-size 256 --chunk-overlap 20"
echo ""

# 24. Skip if collection exists
echo "24. Skip if collection already exists:"
echo "python embedder.py --file document.md --skip-existing"
echo ""

# 25. Verbose output for debugging
echo "25. Verbose mode for debugging:"
echo "python chunker.py --file document.md --type llm --verbose"
echo "python embedder.py --file document.md --verbose"
echo ""

# =============================================================================
# INTEGRATION WITH DIFFERENT LLM PROVIDERS
# =============================================================================

echo "--- LLM PROVIDER EXAMPLES ---"
echo ""

# These examples require appropriate settings in .env file

# 26. Using Gemini (default)
echo "26. Using Gemini for LLM chunking:"
echo "# Set in .env: LLM_PROVIDER=gemini"
echo "python chunker.py --file document.md --type llm"
echo ""

# 27. Using OpenAI
echo "27. Using OpenAI for LLM chunking:"
echo "# Set in .env: LLM_PROVIDER=openai, LLM_MODEL=gpt-4o-mini"
echo "python chunker.py --file document.md --type llm"
echo ""

# 28. Using Ollama (local)
echo "28. Using Ollama local model for LLM chunking:"
echo "# Set in .env: LLM_PROVIDER=ollama, LLM_MODEL=llama3.2"
echo "python chunker.py --file document.md --type llm"
echo ""

# 29. Using LiteLLM
echo "29. Using LiteLLM for multiple providers:"
echo "# Set in .env: LLM_PROVIDER=litellm, LLM_MODEL=openai/gpt-4o-mini"
echo "python chunker.py --file document.md --type llm"
echo ""

# =============================================================================
# TESTING & EVALUATION
# =============================================================================

echo "--- TESTING EXAMPLES ---"
echo ""

# 30. Compare different chunking methods
echo "30. Compare different chunking methods:"
echo "python chunker.py --file document.md --type fixed --output fixed_chunks.json"
echo "python chunker.py --file document.md --type recursive --output recursive_chunks.json"
echo "python chunker.py --file document.md --type llm --output llm_chunks.json"
echo ""

# 31. Test with different chunk sizes
echo "31. Test different chunk sizes:"
echo "python chunker.py --file document.md --chunk-size 256 --output chunks_256.json"
echo "python chunker.py --file document.md --chunk-size 512 --output chunks_512.json"
echo "python chunker.py --file document.md --chunk-size 1024 --output chunks_1024.json"
echo ""

echo "================================"
echo "For more information, see:"
echo "  - SETUP_GUIDE.md"
echo "  - README.md"
echo "  - python chunker.py --help"
echo "  - python embedder.py --help"
echo "================================"
