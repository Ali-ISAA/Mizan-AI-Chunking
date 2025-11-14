# MizanAI Chunking v2.1 - Setup Guide

Complete setup guide for the MizanAI Chunking system.

**Production-Ready:** All 7 chunking strategies are tested, verified, and ready for deployment. See [sample-output/FIXES_VERIFICATION_REPORT.md](sample-output/FIXES_VERIFICATION_REPORT.md) for quality validation details.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Installation](#installation)
3. [Configuration](#configuration)
4. [LLM Provider Setup](#llm-provider-setup)
5. [Vector Store Setup](#vector-store-setup)
6. [Usage Examples](#usage-examples)
7. [Troubleshooting](#troubleshooting)

---

## Quick Start

```bash
# 1. Clone and install
git clone <your-repo-url>
cd Mizan-AI-Chunking
pip install -r requirements.txt

# 2. Configure
cp .env.example .env
# Edit .env and add your API keys

# 3. Run
python chunker.py --file your_document.md
python embedder.py --file your_document.md
```

---

## Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 2: Verify Installation

```bash
python -c "import tiktoken, numpy; print('✓ Core dependencies installed')"
```

---

## Configuration

### Step 1: Create .env File

```bash
cp .env.example .env
```

### Step 2: Configure Providers

Open `.env` and configure at least one LLM provider, one embedding provider, and one vector store:

**Minimum configuration for Gemini + ChromaDB:**
```bash
# LLM
LLM_PROVIDER=gemini
GEMINI_API_KEY_1=your_key_here

# Embeddings
EMBEDDING_PROVIDER=gemini

# Vector Store
VECTOR_STORE=chromadb
CHROMADB_API_KEY=your_key_here
CHROMADB_TENANT=your_tenant_here
CHROMADB_DATABASE=DEV
```

---

## LLM Provider Setup

Choose ONE of the following:

### Option 1: Google Gemini (Recommended)

**Why choose Gemini:**
- Free tier with generous limits
- High-quality semantic analysis
- Automatic API key rotation

**Setup:**
1. Go to https://aistudio.google.com/app/apikey
2. Create API keys (get 2-4 keys for rotation)
3. Add to `.env`:

```bash
LLM_PROVIDER=gemini
LLM_MODEL=gemini-2.0-flash-lite

GEMINI_API_KEY_1=AIzaSy...
GEMINI_API_KEY_2=AIzaSy...
GEMINI_API_KEY_3=AIzaSy...

EMBEDDING_PROVIDER=gemini
EMBEDDING_MODEL=models/embedding-001
EMBEDDING_DIMENSION=768
```

**Rate limits (per key):**
- Embeddings: 100 RPM, 30K TPM, 1K RPD
- Generation: 10 RPM, 250K TPM, 250 RPD
- With 4 keys: 40 RPM generation, 400 RPM embeddings

---

### Option 2: OpenAI

**Setup:**
1. Get API key from https://platform.openai.com/api-keys
2. Add to `.env`:

```bash
LLM_PROVIDER=openai
LLM_MODEL=gpt-4o-mini
OPENAI_API_KEY=sk-...

EMBEDDING_PROVIDER=openai
EMBEDDING_MODEL=text-embedding-3-small
EMBEDDING_DIMENSION=1536
```

**For vLLM or custom OpenAI-compatible endpoints:**
```bash
OPENAI_BASE_URL=http://your-vllm-server:8000/v1
```

---

### Option 3: Ollama (Local/Free)

**Setup:**
1. Install Ollama: https://ollama.com/
2. Pull models:
```bash
ollama pull llama3.2
ollama pull nomic-embed-text
```

3. Add to `.env`:
```bash
LLM_PROVIDER=ollama
LLM_MODEL=llama3.2
OLLAMA_BASE_URL=http://localhost:11434

EMBEDDING_PROVIDER=ollama
EMBEDDING_MODEL=nomic-embed-text
EMBEDDING_DIMENSION=768
```

---

### Option 4: LiteLLM (Multi-Provider)

**Setup:**
```bash
LLM_PROVIDER=litellm
LLM_MODEL=openai/gpt-4o-mini
LITELLM_API_KEY=your_key_here
```

---

## Vector Store Setup

Choose ONE of the following:

### Option 1: ChromaDB (Easiest)

**Cloud (Hosted):**
1. Sign up at https://www.trychroma.com/
2. Create a tenant and database
3. Get API key
4. Add to `.env`:

```bash
VECTOR_STORE=chromadb
CHROMADB_API_KEY=ck-...
CHROMADB_TENANT=xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
CHROMADB_DATABASE=DEV
```

**Local:**
```bash
VECTOR_STORE=chromadb
# No API key needed for local
```

---

### Option 2: Supabase (PostgreSQL + pgvector)

**Setup:**
1. Create project at https://supabase.com/
2. Go to Settings → API
3. Copy URL and service role key
4. Enable pgvector extension:
   - Go to Database → Extensions
   - Enable "vector"
5. Add to `.env`:

```bash
VECTOR_STORE=supabase
SUPABASE_URL=https://xxx.supabase.co
SUPABASE_SERVICE_KEY=eyJhbGci...
```

**Tables are auto-created on first use.**

---

### Option 3: pgvector (Direct PostgreSQL)

**Setup:**
1. Install PostgreSQL with pgvector extension
2. Create database:
```sql
CREATE DATABASE vectordb;
\c vectordb
CREATE EXTENSION vector;
```

3. Add to `.env`:
```bash
VECTOR_STORE=pgvector
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DATABASE=vectordb
POSTGRES_USER=postgres
POSTGRES_PASSWORD=your_password
```

---

### Option 4: Qdrant

**Cloud:**
1. Sign up at https://cloud.qdrant.io/
2. Create cluster
3. Get URL and API key
4. Add to `.env`:

```bash
VECTOR_STORE=qdrant
QDRANT_URL=https://xxx.qdrant.io
QDRANT_API_KEY=your_key
```

**Local (Docker):**
```bash
docker run -p 6333:6333 qdrant/qdrant

VECTOR_STORE=qdrant
QDRANT_URL=http://localhost:6333
```

---

### Option 5: Weaviate

**Cloud:**
1. Sign up at https://console.weaviate.cloud/
2. Create cluster
3. Get URL and API key
4. Add to `.env`:

```bash
VECTOR_STORE=weaviate
WEAVIATE_URL=https://xxx.weaviate.network
WEAVIATE_API_KEY=your_key
```

**Local (Docker):**
```bash
docker run -p 8080:8080 semitechnologies/weaviate:latest

VECTOR_STORE=weaviate
WEAVIATE_URL=http://localhost:8080
```

---

### Option 6: Pinecone

**Setup:**
1. Sign up at https://app.pinecone.io/
2. Create API key
3. Add to `.env`:

```bash
VECTOR_STORE=pinecone
PINECONE_API_KEY=your_key
PINECONE_ENVIRONMENT=us-east-1-aws
PINECONE_INDEX=documents
```

---

## Usage Examples

### Basic Chunking

```bash
# Recursive chunking (recommended default)
python chunker.py --file document.md

# LLM semantic chunking (best quality)
python chunker.py --file document.pdf --type llm

# Fixed token chunking
python chunker.py --file document.txt --type fixed --chunk-size 512 --overlap 50
```

### Embedding and Storage

```bash
# All-in-one: chunk and embed
python embedder.py --file document.md

# Specify chunker and vector store
python embedder.py --file document.pdf --chunker-type llm --vector-store supabase

# Two-step: chunk first, embed later
python chunker.py --file document.md --output chunks.json
python embedder.py --chunks chunks.json --vector-store chromadb
```

### Get Help

```bash
python chunker.py --help
python embedder.py --help
```

---

## Troubleshooting

### "No LLM provider configured"
**Solution:** Add at least one of the following to `.env`:
- `GEMINI_API_KEY_1`
- `OPENAI_API_KEY`
- `OLLAMA_BASE_URL`
- `LITELLM_API_KEY`

### "Resource exhausted (quota)"
**Solution:** You hit API rate limits. For Gemini:
- Add more API keys (`GEMINI_API_KEY_2`, `GEMINI_API_KEY_3`, etc.)
- Wait for quota reset (midnight PST)
- Check usage: https://ai.dev/usage

### "ModuleNotFoundError"
**Solution:** Install missing package:
```bash
pip install <package-name>
```

Or reinstall all:
```bash
pip install -r requirements.txt
```

### "Connection refused" (Ollama)
**Solution:** Start Ollama server:
```bash
ollama serve
```

### "Table does not exist" (Supabase/pgvector)
**Solution:** Tables are auto-created. If this fails:
1. Check database permissions
2. Verify pgvector extension is enabled
3. Check `.env` credentials

### Vector store connection issues
**Solution:**
1. Verify credentials in `.env`
2. Check network connectivity
3. For cloud services, check dashboard status
4. For local services, ensure server is running

---

## Advanced Configuration

### Multiple API Keys for Rate Limit Avoidance

Add up to 10 Gemini keys:
```bash
GEMINI_API_KEY_1=...
GEMINI_API_KEY_2=...
GEMINI_API_KEY_3=...
# ... up to GEMINI_API_KEY_10
```

System automatically rotates on rate limits.

### Custom Chunk Sizes

```bash
# In .env (default for all operations)
CHUNK_SIZE=512
CHUNK_OVERLAP=50

# Or via CLI (per-operation)
python chunker.py --file doc.md --chunk-size 1024 --overlap 100
```

### Custom Collection Names

```bash
# Auto-generated from filename
python embedder.py --file my_document.md
# Creates collection: my_document

# Explicit name
python embedder.py --file doc.md --collection custom_collection_name
```

---

## Verification

Test your setup:

```bash
# 1. Test chunking
echo "This is a test document." > test.txt
python chunker.py --file test.txt --verbose

# 2. Test embedding and storage
python embedder.py --file test.txt --verbose

# 3. Clean up
rm test.txt
```

If both commands succeed, your setup is complete!

---

## Next Steps

1. **Review examples:** `./examples.sh` or `bash examples.sh`
2. **Read documentation:** `README.md` and `CLAUDE.md`
3. **Process your documents:** Start with `python embedder.py --file your_document.md`

## Support

- Check `README.md` for architecture details
- See `examples.sh` for more usage examples
- Review code in `src/` for implementation details
- Open issues on GitHub for bugs/questions
