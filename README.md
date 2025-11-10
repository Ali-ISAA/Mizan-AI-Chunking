# MizanAI Chunking v2.0

Advanced document chunking system with intelligent semantic analysis and multi-provider support. A modern, modular Python framework for document processing, embedding generation, and vector storage.

## Overview

MizanAI Chunking is a comprehensive document processing pipeline that transforms raw documents into semantically meaningful chunks stored in vector databases. The system features a modular architecture with support for multiple LLM providers, embedding services, and vector stores.

**Key Features:**
- 7 different chunking strategies (fixed, recursive, cluster, kamradt, llm, context-aware, section)
- Multiple LLM providers (Gemini, OpenAI, Ollama, LiteLLM)
- Multiple embedding providers (Gemini, OpenAI, Ollama)
- 6 vector stores (ChromaDB, Supabase, pgvector, Qdrant, Weaviate, Pinecone)
- Simple CLI interface with two main commands: `chunker.py` and `embedder.py`
- Automatic API key rotation for rate limit handling
- Modular, extensible architecture

## Project Structure

```
MizanAI-Chunking/
├── src/                       # Core modular architecture (v2.0)
│   ├── chunkers/             # Chunking strategies
│   │   ├── __init__.py       # Chunker registry
│   │   ├── base.py           # Base chunker class
│   │   ├── fixed_token.py    # Fixed-size token chunks
│   │   ├── recursive.py      # Recursive text splitting
│   │   ├── cluster_semantic.py    # Clustering-based semantic chunks
│   │   ├── kamradt_semantic.py    # Similarity-based semantic chunks
│   │   ├── llm_semantic.py        # LLM-based intelligent chunks
│   │   ├── context_aware.py       # Context-preserving chunks
│   │   └── section_based.py       # Section-based splitting
│   │
│   ├── embedders/            # Embedding providers
│   │   ├── __init__.py       # Embedder registry
│   │   ├── base.py           # Base embedder class
│   │   ├── gemini.py         # Google Gemini embeddings
│   │   ├── openai.py         # OpenAI embeddings
│   │   └── ollama.py         # Ollama local embeddings
│   │
│   ├── llms/                 # LLM providers
│   │   ├── __init__.py       # LLM registry
│   │   ├── base.py           # Base LLM class
│   │   ├── gemini.py         # Google Gemini
│   │   ├── openai.py         # OpenAI (GPT models)
│   │   ├── ollama.py         # Ollama local models
│   │   └── litellm.py        # LiteLLM multi-provider
│   │
│   ├── vector_stores/        # Vector database integrations
│   │   ├── __init__.py       # Vector store registry
│   │   ├── base.py           # Base vector store class
│   │   ├── chromadb_store.py      # ChromaDB (local/cloud)
│   │   ├── supabase_store.py      # Supabase with pgvector
│   │   ├── pgvector_store.py      # Direct PostgreSQL + pgvector
│   │   ├── qdrant_store.py        # Qdrant vector database
│   │   ├── weaviate_store.py      # Weaviate vector database
│   │   └── pinecone_store.py      # Pinecone vector database
│   │
│   └── utils/                # Shared utilities
│       ├── __init__.py       # Utility exports
│       ├── config.py         # Configuration management
│       ├── file_reader.py    # File reading utilities
│       └── api_key_manager.py     # API key rotation
│
├── old-files/                # Legacy implementation (v1.0)
│   ├── llm_semantic_chunker/      # Original LLM chunkers
│   ├── other_chunkers/            # Original alternative chunkers
│   └── utils/                     # Original utilities
│
├── chunker.py                # Main CLI for chunking
├── embedder.py               # Main CLI for embedding & storage
├── examples.sh               # Usage examples script
├── .env.example              # Environment template
├── .env                      # Environment variables (git-ignored)
├── requirements.txt          # Python dependencies
├── README.md                 # This file
├── CLAUDE.md                 # Developer guide for Claude Code
└── SETUP_GUIDE.md           # Detailed setup instructions
```

**Excluded from Git** (see [.gitignore](.gitignore)):
- `.env` - API keys and credentials
- `chatlog/`, `ChunkingOutput/`, `Output/` - Generated output folders
- `docs/`, `MD_FILES/`, `tests/` - Working directories
- `*.pdf`, `*.md` (except README.md) - Data files

## Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd Mizan-AI-Chunking

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration

```bash
# Copy environment template
cp .env.example .env

# Edit .env and add your API keys
# At minimum, configure:
# - One LLM provider (for LLM chunking)
# - One embedding provider
# - One vector store
```

**Minimum configuration example (Gemini + ChromaDB):**
```bash
# LLM
LLM_PROVIDER=gemini
GEMINI_API_KEY_1=your_gemini_key_here

# Embeddings
EMBEDDING_PROVIDER=gemini

# Vector Store
VECTOR_STORE=chromadb
CHROMADB_API_KEY=your_chromadb_key_here
CHROMADB_TENANT=your_tenant_id
CHROMADB_DATABASE=DEV
```

See [SETUP_GUIDE.md](SETUP_GUIDE.md) for detailed provider configuration.

### 3. Basic Usage

**Two main commands:**

#### Chunker - Split documents into chunks
```bash
# Basic recursive chunking (recommended default)
python chunker.py --file document.md

# LLM semantic chunking (best quality, requires LLM)
python chunker.py --file document.pdf --type llm

# Save chunks to file for later
python chunker.py --file document.txt --output chunks.json
```

#### Embedder - Generate embeddings and store in vector database
```bash
# All-in-one: chunk, embed, and store
python embedder.py --file document.md

# Specify chunker and vector store
python embedder.py --file document.pdf --chunker-type llm --vector-store supabase

# Load pre-chunked data
python embedder.py --chunks chunks.json --vector-store chromadb
```

### 4. Get Help

```bash
python chunker.py --help
python embedder.py --help

# See more examples
bash examples.sh
```

## Architecture

### Modular Components

The v2.0 architecture is built around four core modules:

#### 1. Chunkers (`src/chunkers/`)
Split documents into meaningful chunks using different strategies.

**Available chunkers:**
- **fixed** - Fixed-size token chunks (simple, predictable)
- **recursive** - Recursive splitting by separators (good default)
- **cluster** - Clustering-based semantic chunks (requires embeddings)
- **kamradt** - Similarity-based semantic chunks (requires embeddings)
- **llm** - LLM-based intelligent semantic chunks (best quality, requires LLM)
- **context-aware** - Markdown-aware with context preservation
- **section** - Split by markdown sections only

All chunkers inherit from `BaseChunker` and implement the `chunk()` method.

#### 2. LLM Providers (`src/llms/`)
Language models for semantic analysis (used by LLM chunker).

**Supported providers:**
- **gemini** - Google Gemini (gemini-2.0-flash-lite)
- **openai** - OpenAI GPT models (gpt-4o-mini, gpt-4, etc.)
- **ollama** - Local Ollama models (llama3.2, mistral, etc.)
- **litellm** - Multi-provider abstraction (supports 100+ models)

#### 3. Embedding Providers (`src/embedders/`)
Generate vector embeddings for semantic search.

**Supported providers:**
- **gemini** - Google Gemini embeddings (models/embedding-001, 768 dimensions)
- **openai** - OpenAI embeddings (text-embedding-3-small, text-embedding-3-large)
- **ollama** - Local Ollama embeddings (nomic-embed-text, etc.)

#### 4. Vector Stores (`src/vector_stores/`)
Store and retrieve embedded chunks.

**Supported stores:**
- **chromadb** - ChromaDB (local or cloud, simplest setup)
- **supabase** - Supabase with pgvector (PostgreSQL-based, auto table creation)
- **pgvector** - Direct PostgreSQL + pgvector (full control)
- **qdrant** - Qdrant vector database (cloud or local)
- **weaviate** - Weaviate vector database (cloud or local)
- **pinecone** - Pinecone vector database (serverless)

All vector stores inherit from `BaseVectorStore` and implement standard operations.

## Key Features

### Modular & Extensible
- Plugin-based architecture for easy additions
- Add new chunkers, embedders, LLMs, or vector stores by extending base classes
- Consistent interfaces across all components

### Multiple Chunking Strategies
- 7 different chunking types for various use cases
- From simple fixed-size to advanced LLM semantic analysis
- Support for markdown, PDF, text, and DOCX files

### Multi-Provider Support
- Switch between providers without code changes
- Use free local models (Ollama) or cloud APIs
- Automatic API key rotation for rate limit handling (Gemini)

### Production-Ready Vector Storage
- 6 vector database integrations
- Automatic collection/table creation
- Batch insertion for efficiency
- Support for local and cloud deployments

### Simple CLI Interface
- Two main commands: `chunker.py` and `embedder.py`
- Intuitive command-line arguments
- Verbose mode for debugging
- JSON output for integration

### Automatic API Key Rotation
- Rotates between multiple API keys automatically (Gemini)
- Avoids rate limits seamlessly
- Failover on quota errors
- Supports up to 10 keys per provider

## Environment Variables

Configuration is managed through the `.env` file. See [SETUP_GUIDE.md](SETUP_GUIDE.md) for detailed setup.

### Core Settings

```bash
# General
CHUNK_SIZE=512                    # Default chunk size in tokens
CHUNK_OVERLAP=50                  # Default overlap in tokens
COLLECTION_NAME=documents         # Default collection name
```

### LLM Providers (choose one)

```bash
# Gemini
LLM_PROVIDER=gemini
LLM_MODEL=gemini-2.0-flash-lite
GEMINI_API_KEY_1=your_key_here
GEMINI_API_KEY_2=your_key_here   # Optional, for rotation

# OpenAI
LLM_PROVIDER=openai
LLM_MODEL=gpt-4o-mini
OPENAI_API_KEY=your_key_here
OPENAI_BASE_URL=https://api.openai.com/v1  # Optional

# Ollama (local)
LLM_PROVIDER=ollama
LLM_MODEL=llama3.2
OLLAMA_BASE_URL=http://localhost:11434

# LiteLLM
LLM_PROVIDER=litellm
LLM_MODEL=openai/gpt-4o-mini
LITELLM_API_KEY=your_key_here
```

### Embedding Providers (choose one)

```bash
# Gemini
EMBEDDING_PROVIDER=gemini
EMBEDDING_MODEL=models/embedding-001
EMBEDDING_DIMENSION=768

# OpenAI
EMBEDDING_PROVIDER=openai
EMBEDDING_MODEL=text-embedding-3-small
EMBEDDING_DIMENSION=1536

# Ollama
EMBEDDING_PROVIDER=ollama
EMBEDDING_MODEL=nomic-embed-text
EMBEDDING_DIMENSION=768
```

### Vector Stores (choose one)

```bash
# ChromaDB
VECTOR_STORE=chromadb
CHROMADB_API_KEY=your_key         # For cloud
CHROMADB_TENANT=your_tenant_id    # For cloud
CHROMADB_DATABASE=DEV             # For cloud

# Supabase
VECTOR_STORE=supabase
SUPABASE_URL=https://xxx.supabase.co
SUPABASE_SERVICE_KEY=your_key

# PostgreSQL + pgvector
VECTOR_STORE=pgvector
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DATABASE=vectordb
POSTGRES_USER=postgres
POSTGRES_PASSWORD=your_password

# Qdrant
VECTOR_STORE=qdrant
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=your_key           # For cloud

# Weaviate
VECTOR_STORE=weaviate
WEAVIATE_URL=http://localhost:8080
WEAVIATE_API_KEY=your_key         # For cloud

# Pinecone
VECTOR_STORE=pinecone
PINECONE_API_KEY=your_key
PINECONE_ENVIRONMENT=us-east-1-aws
PINECONE_INDEX=documents
```

See `.env.example` for complete configuration template.

## Usage Examples

### Basic Chunking

```bash
# Recursive chunking (recommended default)
python chunker.py --file document.md

# LLM semantic chunking (best quality)
python chunker.py --file document.pdf --type llm --chunk-size 512

# Fixed chunking with custom size and overlap
python chunker.py --file document.txt --type fixed --chunk-size 256 --overlap 50

# Context-aware markdown chunking
python chunker.py --file api_docs.md --type context-aware

# Save chunks to JSON for later use
python chunker.py --file document.md --type recursive --output chunks.json --verbose
```

### Embedding and Storage

```bash
# All-in-one: chunk, embed, and store (uses .env settings)
python embedder.py --file document.md

# Specify chunker type and vector store
python embedder.py --file document.pdf --chunker-type llm --vector-store supabase

# Use specific embedding provider
python embedder.py --file document.txt --embedding-provider openai

# Custom collection name
python embedder.py --file document.md --collection my_documents --verbose

# Two-step workflow: chunk first, then embed
python chunker.py --file document.md --type llm --output chunks.json
python embedder.py --chunks chunks.json --vector-store chromadb
```

### Different Vector Stores

```bash
# ChromaDB (cloud or local)
python embedder.py --file document.md --vector-store chromadb

# Supabase (PostgreSQL + pgvector)
python embedder.py --file document.pdf --vector-store supabase

# Qdrant (cloud or local)
python embedder.py --file document.txt --vector-store qdrant

# Pinecone (serverless)
python embedder.py --file document.md --vector-store pinecone --collection docs_index
```

### Advanced Examples

```bash
# Compare different chunking methods
python chunker.py --file doc.md --type fixed --output fixed.json
python chunker.py --file doc.md --type recursive --output recursive.json
python chunker.py --file doc.md --type llm --output llm.json

# Process multiple files
for file in docs/*.md; do
  python embedder.py --file "$file" --vector-store chromadb
done

# Large chunks for code documentation
python embedder.py --file api_docs.md --chunk-size 1024 --chunk-overlap 100

# Small chunks for Q&A
python embedder.py --file faq.txt --chunk-size 256 --chunk-overlap 20
```

See [examples.sh](examples.sh) for more usage examples.

## API Rate Limits

### Google Gemini (Free Tier)
- Embeddings: 100 RPM, 30K TPM, 1K RPD per key
- Generation: 10 RPM, 250K TPM, 250 RPD per key
- **Solution**: Use multiple API keys (GEMINI_API_KEY_1, _2, _3, etc.) for automatic rotation
- With 4 keys: 40 RPM generation, 400 RPM embeddings

### OpenAI
- Varies by tier and model
- Check: https://platform.openai.com/account/rate-limits

### Vector Store Limits
- **ChromaDB Cloud**: Check plan at https://www.trychroma.com/
- **Supabase Free**: 500 MB database, 2 GB bandwidth
- **Qdrant Cloud**: Check plan at https://cloud.qdrant.io/
- **Pinecone**: Check plan at https://www.pinecone.io/pricing/
- **Local stores** (Ollama, local ChromaDB, local Qdrant): No limits

## Troubleshooting

### "No LLM provider configured"
**Solution**: Add at least one LLM provider to `.env`:
- For Gemini: `GEMINI_API_KEY_1=your_key`
- For OpenAI: `OPENAI_API_KEY=your_key`
- For Ollama: `OLLAMA_BASE_URL=http://localhost:11434`

### "Resource exhausted (quota)" (Gemini)
**Solution**:
- Add more API keys for rotation: `GEMINI_API_KEY_2`, `GEMINI_API_KEY_3`, etc.
- System automatically rotates on rate limits
- Or wait for quota reset (midnight PST)

### "ModuleNotFoundError"
**Solution**: Install missing dependencies:
```bash
pip install -r requirements.txt
```

### "Connection refused" (Ollama)
**Solution**: Start Ollama server:
```bash
ollama serve
```

### "Table does not exist" (Supabase/pgvector)
**Solution**:
- Tables are auto-created on first use
- Check database permissions and pgvector extension is enabled
- Verify `.env` credentials

### Vector store connection issues
**Solution**:
1. Verify credentials in `.env`
2. Check network connectivity
3. For cloud services, check dashboard status
4. For local services, ensure server is running

See [SETUP_GUIDE.md](SETUP_GUIDE.md) for detailed troubleshooting.

## Extending the System

### Adding a New Chunker

1. Create file in `src/chunkers/your_chunker.py`
2. Inherit from `BaseChunker`
3. Implement `chunk()` method
4. Register in `src/chunkers/__init__.py`

```python
from .base import BaseChunker

class YourChunker(BaseChunker):
    def chunk(self, text: str, metadata: dict = None) -> List[dict]:
        # Your chunking logic here
        return chunks
```

### Adding a New LLM Provider

1. Create file in `src/llms/your_llm.py`
2. Inherit from `BaseLLM`
3. Implement `generate()` method
4. Register in `src/llms/__init__.py`

### Adding a New Vector Store

1. Create file in `src/vector_stores/your_store.py`
2. Inherit from `BaseVectorStore`
3. Implement required methods
4. Register in `src/vector_stores/__init__.py`

See existing implementations for examples.

## Legacy Implementation

The original v1.0 implementation is preserved in the `old-files/` directory:
- `old-files/llm_semantic_chunker/` - Original LLM chunkers
- `old-files/other_chunkers/` - Original alternative methods
- `old-files/utils/` - Original utilities

These files are kept for reference but are no longer maintained.

## Documentation

- **[README.md](README.md)** - This file, project overview
- **[SETUP_GUIDE.md](SETUP_GUIDE.md)** - Detailed setup instructions for all providers
- **[CLAUDE.md](CLAUDE.md)** - Developer guide for working with Claude Code
- **[examples.sh](examples.sh)** - Comprehensive usage examples
- **[.env.example](.env.example)** - Environment variable template

## Contributing

Contributions are welcome! To add support for new providers:

1. Fork the repository
2. Create a feature branch
3. Follow the extension patterns (see "Extending the System")
4. Never commit API keys or `.env` files
5. Test with multiple providers
6. Submit a pull request

## Support

**For setup help:**
- See [SETUP_GUIDE.md](SETUP_GUIDE.md) for detailed configuration
- Run `python chunker.py --help` or `python embedder.py --help`
- Check [examples.sh](examples.sh) for usage patterns

**For issues:**
- Check troubleshooting section above
- Review [CLAUDE.md](CLAUDE.md) for architecture details
- Open an issue on GitHub

## Acknowledgments

- **Google Gemini** - LLM and embedding models
- **OpenAI** - GPT models and embeddings
- **ChromaDB** - Vector database
- **Supabase** - PostgreSQL + pgvector
- **Qdrant, Weaviate, Pinecone** - Vector database solutions
- **Ollama** - Local LLM inference
- **LangChain** - Inspiration for semantic chunking approaches
