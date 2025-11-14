# MizanAI Chunking v2.1

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

# Batch process directory
python chunker.py --dir sample-md-files --output-dir chunks_output
```

#### Embedder - Generate embeddings and store in vector database
```bash
# All-in-one: chunk, embed, and store
python embedder.py --file document.md

# Specify chunker and vector store
python embedder.py --file document.pdf --chunker-type llm --vector-store supabase

# Load pre-chunked data
python embedder.py --chunks chunks.json --vector-store chromadb

# Batch process directory of chunk files (all go into same collection)
python embedder.py --dir chunks_output --vector-store chromadb
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

## Chunking Strategies Explained

MizanAI Chunking provides **7 different chunking strategies** to handle various document types and use cases. Each strategy has its own strengths and ideal scenarios.

### 1. Fixed Token Chunker (`fixed`)

**How it works:**
- Splits text into equal-sized chunks based on token count
- Simple sliding window approach with configurable overlap
- No semantic analysis, purely mechanical splitting

**When to use:**
- Quick prototyping and testing
- When you need predictable, uniform chunk sizes
- Simple documents without complex structure
- When speed is critical and semantic coherence is less important

**Process:**
1. Tokenize entire document
2. Split into chunks of exactly N tokens
3. Apply overlap between consecutive chunks

**Speed:** ⚡⚡⚡ **Very Fast** (milliseconds for large documents)

**Example:**
```bash
python chunker.py --file document.txt --type fixed --chunk-size 256 --overlap 50
```

---

### 2. Recursive Chunker (`recursive`) ⭐ **Recommended Default**

**How it works:**
- Recursively splits text using a hierarchy of separators
- Tries to split on paragraphs first, then sentences, then words
- Respects natural text boundaries while maintaining target chunk size
- Balances semantic coherence with size constraints

**When to use:**
- General-purpose chunking for most documents
- When you want good results without complexity
- Markdown, text, and structured documents
- Best balance of speed and quality

**Process:**
1. Try to split on paragraph breaks (`\n\n`)
2. If chunks too large, split on sentences (`.`, `!`, `?`)
3. If still too large, split on clauses (`,`, `;`)
4. Finally split on words or characters
5. Merge small chunks to meet target size

**Speed:** ⚡⚡⚡ **Very Fast** (seconds for large documents)

**Example:**
```bash
python chunker.py --file document.md --type recursive --chunk-size 512
```

---

### 3. Cluster Semantic Chunker (`cluster`)

**How it works:**
- Uses embedding-based clustering to group semantically similar sentences
- Applies K-means clustering on sentence embeddings
- Creates chunks by grouping sentences in the same cluster

**When to use:**
- Documents with mixed topics that need semantic separation
- When you want to group related content regardless of position
- Research papers, articles with multiple distinct sections
- When semantic coherence is more important than position

**Process:**
1. Split text into sentences
2. **Generate embeddings** for each sentence using configured embedding provider
3. Apply K-means clustering (auto-determines optimal K or uses specified `--num-clusters`)
4. Group consecutive sentences belonging to same cluster
5. Form chunks from these groups
6. Ensure chunks meet size constraints

**Speed:** ⚡ **Moderate to Slow** (requires embedding generation for all sentences)
- Small docs (<5K tokens): ~10-30 seconds
- Large docs (>20K tokens): 1-3 minutes
- **Note:** Speed depends on embedding provider and rate limits

**Example:**
```bash
# Auto-determine number of clusters
python chunker.py --file document.txt --type cluster --chunk-size 512

# Specify number of clusters
python chunker.py --file document.md --type cluster --num-clusters 15
```

**Requirements:**
- Configured embedding provider in `.env`
- Will use the embedding provider specified (Gemini, OpenAI, or Ollama)

---

### 4. Kamradt Semantic Chunker (`kamradt`)

**How it works:**
- Based on Greg Kamradt's semantic chunking approach
- Calculates similarity between consecutive sentences using embeddings
- Splits at points where similarity drops significantly (breakpoints)
- Creates variable-sized chunks based on semantic coherence

**When to use:**
- Long-form content with natural topic transitions
- Articles, blog posts, documentation
- When you want chunks that align with natural topic changes
- Better than recursive for documents with clear semantic shifts

**Process:**
1. Split text into sentences
2. **Generate embeddings** for each sentence
3. Calculate cosine similarity between consecutive sentences
4. Identify breakpoints where similarity falls below percentile threshold
5. Split document at breakpoints to create chunks
6. Merge small chunks to meet minimum size

**Speed:** ⚡ **Moderate to Slow** (requires embedding generation)
- Small docs: ~10-30 seconds
- Large docs: 1-3 minutes
- Similar performance to cluster chunker

**Example:**
```bash
# Default: 95th percentile threshold
python chunker.py --file document.md --type kamradt

# Custom breakpoint threshold (lower = more splits)
python chunker.py --file document.txt --type kamradt --breakpoint-percentile 90
```

**Requirements:**
- Configured embedding provider in `.env`

---

### 5. LLM Semantic Chunker (`llm`) ⭐ **Best Quality**

**How it works:**
- Uses a language model to intelligently analyze and split text
- LLM identifies natural breakpoints based on semantic meaning
- Creates chunks that preserve complete thoughts and context
- Most sophisticated chunking method

**When to use:**
- High-stakes applications where quality matters most
- Complex documents with nuanced structure
- When you need chunks that make sense to humans
- Technical documentation, legal documents, research papers
- When you have LLM quota to spare

**Process:**
1. Send text to LLM with chunking instructions
2. LLM analyzes semantic structure and topic boundaries
3. LLM proposes chunk boundaries with reasoning
4. System splits text at LLM-suggested points
5. Validates chunk sizes and adjusts if needed

**Speed:** 🐌 **Slow** (requires LLM API calls)
- Small docs (<5K tokens): ~20-60 seconds
- Large docs (>20K tokens): 2-10 minutes
- **Note:** Speed depends on LLM provider, model, and rate limits

**Example:**
```bash
# Default: uses LLM from .env
python chunker.py --file document.pdf --type llm --chunk-size 512

# With verbose output to see LLM reasoning
python chunker.py --file document.md --type llm --verbose
```

**Requirements:**
- Configured LLM provider in `.env` (Gemini, OpenAI, Ollama, or LiteLLM)
- Consumes LLM API quota (approximately 2-5 requests per document)

**Cost Considerations:**
- **Gemini (gemini-2.0-flash-lite)**: Free tier, 10 RPM limit
- **OpenAI (gpt-4o-mini)**: ~$0.15-0.60 per 1M tokens
- **Ollama**: Free (local), but requires local GPU/CPU

---

### 6. Context-Aware Chunker (`context-aware`)

**How it works:**
- Markdown-aware chunking that preserves document structure
- Keeps hierarchical context (headers, lists, code blocks)
- Maintains parent headers in chunk metadata
- Respects markdown boundaries (code blocks, tables, lists)

**When to use:**
- Markdown documentation and wikis
- API documentation, README files
- When preserving document structure is important
- Technical documentation with code examples

**Process:**
1. Parse markdown structure (headers, code blocks, lists, etc.)
2. Identify semantic units (sections, subsections)
3. Create chunks that respect markdown boundaries
4. Include parent header context in metadata
5. Never split code blocks or tables mid-content

**Speed:** ⚡⚡ **Fast** (no external API calls, intelligent parsing)
- Small docs: <1 second
- Large docs: 1-5 seconds

**Example:**
```bash
python chunker.py --file api_docs.md --type context-aware
```

**Output includes:**
- Full chunk text with proper markdown formatting
- Metadata with parent headers (e.g., `# Overview > ## Features`)
- Preserved code blocks and tables

---

### 7. Section-Based Chunker (`section`)

**How it works:**
- Simplest semantic approach: split only at markdown headers
- Each chunk is one complete section
- No token-size enforcement (sections can vary widely in size)
- Ideal for structured documents with clear sections

**When to use:**
- Well-structured markdown with clear sections
- When each section should be treated as atomic unit
- Documentation where sections are self-contained
- When section boundaries are more important than size uniformity

**Process:**
1. Parse markdown headers (`#`, `##`, `###`, etc.)
2. Split document at header boundaries
3. Each section becomes one chunk (regardless of size)
4. Preserve header hierarchy in metadata

**Speed:** ⚡⚡⚡ **Very Fast** (simple regex parsing)

**Example:**
```bash
python chunker.py --file documentation.md --type section
```

**Note:** Chunk sizes will vary widely (100-5000+ tokens per chunk)

---

### Comparison Table

| Strategy | Speed | Quality | Semantic Aware | Needs Embeddings | Needs LLM | Best For |
|----------|-------|---------|----------------|------------------|-----------|----------|
| **fixed** | ⚡⚡⚡ Very Fast | ⭐ Basic | ❌ No | ❌ No | ❌ No | Quick tests, uniform sizes |
| **recursive** | ⚡⚡⚡ Very Fast | ⭐⭐⭐ Good | ✅ Partial | ❌ No | ❌ No | **General purpose** (recommended) |
| **cluster** | ⚡ Moderate | ⭐⭐⭐⭐ Very Good | ✅ Yes | ✅ Yes | ❌ No | Mixed-topic documents |
| **kamradt** | ⚡ Moderate | ⭐⭐⭐⭐ Very Good | ✅ Yes | ✅ Yes | ❌ No | Long-form content with topic shifts |
| **llm** | 🐌 Slow | ⭐⭐⭐⭐⭐ Excellent | ✅ Yes | ❌ No | ✅ Yes | **Highest quality** (best results) |
| **context-aware** | ⚡⚡ Fast | ⭐⭐⭐⭐ Very Good | ✅ Yes | ❌ No | ❌ No | Markdown documentation |
| **section** | ⚡⚡⚡ Very Fast | ⭐⭐⭐ Good | ✅ Partial | ❌ No | ❌ No | Structured docs with sections |

### Speed Details

**Very Fast (⚡⚡⚡):** < 1 second for most documents
- fixed, recursive, section

**Fast (⚡⚡):** 1-5 seconds for most documents
- context-aware

**Moderate (⚡):** 10 seconds to 3 minutes (depends on doc size and API rate limits)
- cluster, kamradt
- Speed limited by embedding generation
- With 4 Gemini API keys: ~400 RPM (can process ~100 chunks/minute)

**Slow (🐌):** 20 seconds to 10+ minutes (depends on doc size and LLM speed)
- llm
- Speed limited by LLM generation rate
- With 4 Gemini API keys: ~40 RPM (slower but higher quality)

### Choosing the Right Strategy

**Quick Decision Guide:**

1. **Need fast prototyping?** → Use `recursive` (recommended default)
2. **Need best quality and have time?** → Use `llm`
3. **Working with markdown docs?** → Use `context-aware` or `section`
4. **Document has distinct topics?** → Use `cluster` or `kamradt`
5. **Need uniform sizes quickly?** → Use `fixed`

**Performance Tiers:**

- **Free tier + fast:** `recursive`, `context-aware`, `section`, `fixed`
- **Free tier + quality (moderate speed):** `cluster`, `kamradt` (needs embeddings)
- **Best quality (slower + API costs):** `llm` (needs LLM provider)

---

## Collection Strategy & Vector Store Organization

Understanding how to organize your documents in vector stores is crucial for effective semantic search and RAG applications.

### What are Collections?

Collections are logical groupings of vectors in a vector database. Different vector stores use different terminology but serve the same purpose:

| Vector Store | Terminology | Description |
|--------------|-------------|-------------|
| **Qdrant** | Collection | Logical group of vectors with metadata |
| **ChromaDB** | Collection | Named group of embeddings |
| **Supabase/pgvector** | Table | PostgreSQL table with vector column |
| **Weaviate** | Class | Schema-based data object type |
| **Pinecone** | Index | Top-level namespace for vectors |

### Strategy 1: Single Collection (Recommended for Most Use Cases) ⭐

**Store all documents in ONE collection, use metadata to distinguish them.**

```bash
# All documents go to one collection
python embedder.py --dir sample-output/llm --vector-store qdrant --collection knowledge_base
python embedder.py --dir sample-output/recursive --vector-store qdrant --collection knowledge_base
python embedder.py --dir sample-output/context-aware --vector-store qdrant --collection knowledge_base
```

**Benefits:**
- ✅ **Semantic search across ALL documents** (the main benefit!)
- ✅ Find related information across different documents
- ✅ Simpler management - one place to search
- ✅ Still filterable by document using metadata
- ✅ Perfect for RAG applications and chatbots
- ✅ Better results when topics are related

**When to use:**
- Company knowledge base (search all documents together)
- Documentation site (search across all guides)
- Customer support articles
- RAG chatbot (search across all context)
- Research papers library
- Code documentation
- Multi-document Q&A systems

**How metadata filtering works:**

Each chunk automatically includes metadata:
```json
{
  "text": "The company's return policy allows...",
  "metadata": {
    "source_file": "customer_policy.pdf",
    "chunk_index": 5,
    "chunker_type": "llm",
    "document_id": "policy_001",
    "department": "customer_service"
  }
}
```

Search with filters (vector store dependent):
```python
# Search only in specific document
results = vector_store.search(
    query_embedding,
    top_k=5,
    filter={"source_file": "customer_policy.pdf"}
)

# Search only LLM-chunked content
results = vector_store.search(
    query_embedding,
    top_k=5,
    filter={"chunker_type": "llm"}
)

# Search specific department
results = vector_store.search(
    query_embedding,
    top_k=5,
    filter={"department": "customer_service"}
)
```

### Strategy 2: Multiple Collections (Use for Data Isolation)

**Separate collection per project/tenant/domain when you need complete isolation.**

```bash
# Customer A's data (completely isolated)
python embedder.py --dir customer_a_docs --vector-store qdrant --collection customer_a

# Customer B's data (completely isolated)
python embedder.py --dir customer_b_docs --vector-store qdrant --collection customer_b

# Legal documents (separate domain)
python embedder.py --dir legal_docs --vector-store qdrant --collection legal_docs

# Medical documents (separate domain)
python embedder.py --dir medical_docs --vector-store qdrant --collection medical_docs
```

**Benefits:**
- ✅ Complete data isolation (security requirement)
- ✅ Different configurations per collection
- ✅ Can use different embedding models
- ✅ Independent scaling, backup, and deletion
- ✅ Access control at collection level

**When to use:**
- **Multi-tenant SaaS** - One collection per customer (critical for security)
- **Different embedding models** - Different domains need different embeddings
- **Access control** - Different teams/permissions per collection
- **Completely separate domains** - Legal vs medical vs technical (no semantic overlap)
- **Different languages** - One collection per language
- **Compliance requirements** - Data must be physically/logically separated

### Practical Examples

#### Example 1: Company Knowledge Base (Single Collection)

```bash
# All company docs in one searchable collection
python embedder.py --file hr_handbook.pdf --collection company_kb --vector-store qdrant
python embedder.py --file product_guide.pdf --collection company_kb --vector-store qdrant
python embedder.py --file return_policy.pdf --collection company_kb --vector-store qdrant

# Search query: "What is our return policy?"
# ✅ Can find info across: HR docs, product docs, policy docs
# ✅ Better results because related content is searchable together
```

#### Example 2: Multi-Tenant SaaS (Multiple Collections)

```bash
# Each customer gets their own isolated collection
python embedder.py --dir tenant_123_docs --collection tenant_123 --vector-store qdrant
python embedder.py --dir tenant_456_docs --collection tenant_456 --vector-store qdrant

# Customer 123 can ONLY search collection: tenant_123
# Customer 456 can ONLY search collection: tenant_456
# ✅ Complete data isolation for security and compliance
```

#### Example 3: Testing Different Chunking Strategies (Single Collection)

```bash
# Compare different chunkers in same collection
python embedder.py --dir sample-output/llm --collection test_all --vector-store qdrant
python embedder.py --dir sample-output/recursive --collection test_all --vector-store qdrant
python embedder.py --dir sample-output/cluster --collection test_all --vector-store qdrant

# Metadata will show: chunker_type: "llm", "recursive", "cluster"
# ✅ Easy to compare which chunker gives better search results!
```

### Vector Store Capabilities Comparison

| Feature | Qdrant | ChromaDB | Supabase/pgvector | Weaviate | Pinecone |
|---------|--------|----------|-------------------|----------|----------|
| **Multiple Collections** | ✅ Unlimited | ✅ Unlimited | ✅ Many tables | ✅ Many classes | ✅ Limited (paid) |
| **Metadata Filtering** | ✅ Rich filters | ✅ Basic WHERE | ✅ Full SQL | ✅ GraphQL | ✅ Rich filters |
| **Collection Limits** | No practical limit | No practical limit | DB table limit | No limit | Plan dependent |
| **Cross-Collection Search** | ❌ Not supported | ❌ Not supported | ✅ SQL JOINs | ✅ Cross-refs | ❌ Not supported |
| **Dynamic Schema** | ✅ Flexible | ✅ Flexible | ⚠️ Table schema | ✅ Flexible | ✅ Flexible |
| **Metadata Types** | JSON objects | Dict/JSON | PostgreSQL types | GraphQL types | JSON metadata |

### Best Practices

#### ✅ DO:
- **Use single collection for related documents** - Better semantic search and discovery
- **Use metadata extensively** - Filter without creating multiple collections
- **Name collections descriptively** - `customer_docs` not `collection_1`
- **Use multiple collections for tenants** - Security and compliance requirement
- **Test chunking strategies in same collection** - Easy A/B comparison
- **Plan metadata schema early** - Consistent metadata enables better filtering

#### ❌ DON'T:
- **Don't create collection per document** - Loses cross-document semantic search
- **Don't use collections for simple filtering** - Use metadata instead
- **Don't mix different embedding models in one collection** - Vectors won't be comparable
- **Don't over-complicate** - Start with single collection, split only when needed
- **Don't forget to add custom metadata** - Source, category, date, etc.

### Decision Tree

```
┌─ Do you need complete data isolation (multi-tenant, security)?
│
├─ YES → Multiple Collections
│         ├─ One collection per tenant/customer
│         ├─ One collection per compliance domain
│         └─ One collection per language
│
└─ NO  → Do you want to search ACROSS all documents?
          │
          ├─ YES → Single Collection + Metadata filtering
          │         ├─ Store all documents together
          │         ├─ Use metadata for filtering
          │         └─ Better semantic search results
          │
          └─ NO  → Are these documents related at all?
                   ├─ YES → You probably want single collection
                   └─ NO  → Multiple collections (different domains)
```

### Collection Naming Conventions

**Good names:**
```bash
--collection company_knowledge_base
--collection customer_support_articles
--collection product_documentation
--collection tenant_acme_corp
--collection legal_compliance_docs
```

**Avoid:**
```bash
--collection collection1          # Too generic
--collection test                 # Not descriptive
--collection my_collection        # Unclear purpose
--collection data                 # Too vague
```

### When to Refactor

**Signs you need to split collections:**
1. Different teams need different access permissions
2. Embedding model changes for a subset of documents
3. Multi-tenant requirements emerge
4. Compliance requires data isolation
5. Performance degrades due to collection size (rare with modern vector DBs)

**Signs you should merge collections:**
1. Frequently searching across multiple collections
2. Related documents in different collections
3. Duplicate metadata management
4. Complicated application logic to handle multiple collections

---

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

# Batch workflow: process directory of files
python chunker.py --dir sample-md-files --output-dir chunks_output
python embedder.py --dir chunks_output --vector-store chromadb
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

# Process multiple files (recommended: use batch processing)
python chunker.py --dir docs --output-dir chunks_output
python embedder.py --dir chunks_output --vector-store chromadb

# Alternative: process files one by one
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
