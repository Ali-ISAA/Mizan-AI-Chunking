# MizanAiChunking

Advanced document chunking system with LLM-based semantic analysis and RAG chatbot capabilities. Supports both ChromaDB and Supabase vector databases with hybrid search (semantic + keyword).

## Overview

This project provides multiple intelligent document chunking methods, with a focus on LLM-based semantic chunking using Google Gemini. It includes a complete RAG (Retrieval-Augmented Generation) chatbot system with hybrid search capabilities.

## Project Structure

```
MizanAiChunking/
├── llm_semantic_chunker/     # LLM-based semantic chunking (PRIMARY)
│   ├── llm_semantic_md_chunker.py           # ChromaDB version
│   ├── llm_semantic_md_chunker_supabase.py  # Supabase version
│   ├── llm_semantic_pdf_chunker.py          # PDF chunker
│   ├── api_key_manager.py                   # API key rotation
│   ├── file_reader.py                       # File utilities
│   ├── supabase_setup_function.sql          # One-time SQL setup
│   ├── SETUP_ONCE.md                        # Supabase setup guide
│   └── README_SUPABASE.md                   # Supabase documentation
├── other_chunkers/            # Alternative chunking methods
│   ├── context_aware_md_chunker.py
│   ├── context_aware_pdf_chunker.py
│   ├── markdown_section_chunker.py
│   └── text_chunking_methods.py
├── utils/                     # Utility scripts
│   ├── reset_chromadb_collection.py
│   └── disable_embeddings_patch.py
├── .env                       # Environment variables (git-ignored)
├── .gitignore                 # Git exclusions
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

**Excluded from Git** (see [.gitignore](.gitignore)):
- `chatlog/` - Chat conversation logs (excluded)
- `ChunkingOutput/`, `Output/` - Generated output folders
- `docs/`, `MD_FILES/`, `tests/`, `chunking_evaluation/`, `BaseFileForMD/` - Excluded project folders
- `*.pdf`, `*.md` (except README.md) - Data files
- `.env` - Secrets and API keys

## Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/MizanAiChunking.git
cd MizanAiChunking

# Install dependencies
pip install -r requirements.txt
```

### 2. Environment Configuration

Create a `.env` file in the root directory with the following structure:

```bash
# =============================================================================
# GOOGLE GEMINI API KEYS (For LLM and Embeddings)
# =============================================================================
# Get your keys from: https://aistudio.google.com/app/apikey
# The system uses automatic rotation between multiple keys to avoid rate limits

GEMINI_API_KEY_1=AIzaSyXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
GEMINI_API_KEY_2=AIzaSyXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
GEMINI_API_KEY_3=AIzaSyXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
GEMINI_API_KEY_4=AIzaSyXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX

# =============================================================================
# CHROMADB CLOUD CREDENTIALS (For Vector Storage)
# =============================================================================
# Get your credentials from: https://www.trychroma.com/
# Used by: llm_semantic_md_chunker.py

CHROMADB_API_KEY=ck-XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
CHROMADB_TENANT=xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
CHROMADB_DATABASE=DEV

# =============================================================================
# SUPABASE CREDENTIALS (Alternative Vector Storage with PostgreSQL + pgvector)
# =============================================================================
# Get your credentials from: https://supabase.com/dashboard/project/_/settings/api
# Used by: llm_semantic_md_chunker_supabase.py

SUPABASE_URL=https://xxxxxxxxxxxxx.supabase.co
SUPABASE_ANON_KEY=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
SUPABASE_SERVICE_KEY=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
```

**Important Security Notes:**
- ⚠️ Never commit the `.env` file to Git (it's already in `.gitignore`)
- ⚠️ Never hardcode API keys in your Python files
- ⚠️ Use environment variables for all credentials
- ✅ The `.env` file is automatically loaded by all scripts

### 3. Choose Your Vector Database

You can use either **ChromaDB** (cloud-based) or **Supabase** (PostgreSQL + pgvector):

#### Option A: ChromaDB (Simpler Setup)

```bash
cd llm_semantic_chunker
python llm_semantic_md_chunker.py

# The script will:
# 1. List all .md files in MD_FILES/ folder
# 2. Let you select which file to process
# 3. Perform LLM semantic analysis
# 4. Save chunks to ChromaDB Cloud
```

#### Option B: Supabase (More Control, PostgreSQL-based)

**One-time setup required:**

1. Go to your Supabase Dashboard → SQL Editor
2. Run the SQL from `llm_semantic_chunker/supabase_setup_function.sql`
3. This creates a function for automatic table creation

For detailed setup instructions, see: [llm_semantic_chunker/SETUP_ONCE.md](llm_semantic_chunker/SETUP_ONCE.md)

```bash
cd llm_semantic_chunker
python llm_semantic_md_chunker_supabase.py

# The script will:
# 1. List all .md files in MD_FILES/ folder
# 2. Let you select which file to process
# 3. Perform LLM semantic analysis
# 4. Automatically create Supabase table (if needed)
# 5. Save chunks with embeddings to Supabase
```

### 4. Place Your Documents

```bash
# Create MD_FILES folder if it doesn't exist
mkdir MD_FILES

# Place your .md files there
cp your_document.md MD_FILES/
```

## Modules

### 📁 llm_semantic_chunker/
**LLM-Based Semantic Chunking** - Uses Google Gemini to intelligently chunk documents

**Main Scripts:**
- `llm_semantic_md_chunker.py` - **ChromaDB version** for cloud vector storage
- `llm_semantic_md_chunker_supabase.py` - **Supabase version** for PostgreSQL + pgvector
- `llm_semantic_pdf_chunker.py` - PDF semantic chunker
- `api_key_manager.py` - Automatic API key rotation to avoid rate limits
- `file_reader.py` - File reading utilities

**Supabase Setup Files:**
- `supabase_setup_function.sql` - SQL for one-time Supabase setup
- `SETUP_ONCE.md` - Step-by-step Supabase configuration guide
- `README_SUPABASE.md` - Complete Supabase documentation

**Features:**
- Pure semantic analysis (no arbitrary size limits)
- Automatic API key rotation across 4 keys
- Dual database support (ChromaDB Cloud + Supabase)
- Automatic table creation for Supabase
- Post-processing for oversized chunks
- Batch insertion (100 records per batch for Supabase)

**How it works:**
1. Reads markdown file
2. Sends text to Gemini LLM for semantic boundary detection
3. Generates embeddings using Gemini embedding-001 (768 dimensions)
4. Stores chunks + embeddings in ChromaDB or Supabase
5. Ready for RAG retrieval

### 📁 other_chunkers/
**Alternative Chunking Methods**

- `context_aware_md_chunker.py` - Context-aware markdown chunking
- `context_aware_pdf_chunker.py` - Context-aware PDF chunking
- `markdown_section_chunker.py` - Section-based markdown chunking
- `text_chunking_methods.py` - 5 traditional methods:
  1. Fixed Token Chunking
  2. Recursive Token Chunking
  3. Cluster Semantic Chunking
  4. Kamradt Semantic Chunking
  5. LLM Semantic Chunking

All methods support optional embedding generation to avoid quota limits.

### 📁 utils/
**Utility Scripts**

- `reset_chromadb_collection.py` - Delete and recreate ChromaDB collections with fresh chunks
- `disable_embeddings_patch.py` - Patch to disable embeddings by default in chunking methods

**Security:** All utilities now use environment variables instead of hardcoded credentials.

## Key Features

### 🧠 LLM Semantic Chunking
- Uses Gemini AI to understand content semantics
- Creates meaningful chunks based on topic boundaries
- No arbitrary size limits (pure semantic approach)

### 🔄 Automatic API Key Rotation
- Rotates between 4 Gemini API keys automatically
- Avoids rate limits (15 RPM per key = 60 RPM total)
- Seamless failover on quota errors

### 🗄️ Dual Vector Database Support
- **ChromaDB Cloud**: Managed cloud service, zero setup
- **Supabase**: PostgreSQL + pgvector, full control, automatic table creation

### 🔍 Hybrid Search (Chatbot - Excluded from Git)
- Semantic search using embeddings (cosine similarity)
- Keyword search using BM25 algorithm
- Reciprocal Rank Fusion (RRF) for result merging
- Retrieves 100 chunks for comprehensive context

### 📊 Post-Processing
- Automatically splits oversized chunks (>2000 tokens)
- Maintains semantic boundaries during splitting
- Ensures optimal chunk sizes for RAG

### 🔐 Security
- All credentials in `.env` file (git-ignored)
- No hardcoded API keys in codebase
- Environment variable validation
- Secure credential loading with `python-dotenv`

## Environment Variables Reference

| Variable | Purpose | Required For | Where to Get |
|----------|---------|--------------|--------------|
| `GEMINI_API_KEY_1` to `GEMINI_API_KEY_4` | Google Gemini LLM & Embeddings | All chunkers | https://aistudio.google.com/app/apikey |
| `CHROMADB_API_KEY` | ChromaDB authentication | ChromaDB version | https://www.trychroma.com/ |
| `CHROMADB_TENANT` | ChromaDB tenant ID | ChromaDB version | ChromaDB dashboard |
| `CHROMADB_DATABASE` | ChromaDB database name | ChromaDB version | ChromaDB dashboard |
| `SUPABASE_URL` | Supabase project URL | Supabase version | Supabase project settings |
| `SUPABASE_ANON_KEY` | Supabase anonymous key | Supabase version | Supabase API settings |
| `SUPABASE_SERVICE_KEY` | Supabase service role key | Supabase version | Supabase API settings |

## Supabase Setup

For Supabase integration, you need to run a **one-time SQL setup** to enable automatic table creation:

1. Open Supabase Dashboard → SQL Editor
2. Copy SQL from `llm_semantic_chunker/supabase_setup_function.sql`
3. Run the SQL (creates `create_document_table()` function)
4. Done! Python script will auto-create tables for each document

**Detailed guide:** [llm_semantic_chunker/SETUP_ONCE.md](llm_semantic_chunker/SETUP_ONCE.md)

**Supabase features:**
- Automatic table creation per document
- Vector similarity search (pgvector with IVFFlat index)
- PostgreSQL-based (full SQL access)
- Batch insertion (100 records per batch)
- 768-dimension embeddings from Gemini

## Usage Examples

### Process a Markdown File (ChromaDB)

```bash
cd llm_semantic_chunker
python llm_semantic_md_chunker.py
```

**Output:**
```
Available markdown files:
1. digital_government_policies.md
2. n8n_documentation.md

Select file number: 2

Processing: n8n_documentation.md
Generating semantic chunks...
Created 245 chunks
Generating embeddings...
Stored in ChromaDB collection: n8n_documentation_md_processed
```

### Process a Markdown File (Supabase)

```bash
cd llm_semantic_chunker
python llm_semantic_md_chunker_supabase.py
```

**Output:**
```
Available markdown files:
1. digital_government_policies.md
2. n8n_documentation.md

Select file number: 2

Processing: n8n_documentation.md
Generating semantic chunks...
Created 245 chunks
Checking if table exists...
Table 'doc_n8n_documentation_md_processed' created
Generating embeddings...
Inserting batch 1/3 (100 records)...
Inserting batch 2/3 (100 records)...
Inserting batch 3/3 (45 records)...
Stored 245 chunks in Supabase
```

### Reset ChromaDB Collection

```bash
cd utils
python reset_chromadb_collection.py
```

This will:
1. Delete old collection
2. Generate fresh chunks from PDF
3. Filter out page headers and short chunks
4. Create new collection
5. Test with sample query

## API Rate Limits

**Google Gemini API** (Free Tier):
- 15 requests per minute (RPM) per API key
- 1,500 requests per day (RPD) per API key
- Solution: We use 4 keys in rotation = **60 RPM total**

**ChromaDB Cloud** (Free Tier):
- Check your plan limits at https://www.trychroma.com/

**Supabase** (Free Tier):
- 500 MB database
- 50,000 monthly active users
- 2 GB bandwidth
- Check limits at https://supabase.com/pricing

## Troubleshooting

### "Resource has been exhausted (e.g. check quota)"
- You've hit the API rate limit
- Solution: The system automatically rotates to the next API key
- Add more keys to `.env` if needed

### "Could not find the function create_document_table"
- You haven't run the one-time Supabase SQL setup
- Solution: Follow [llm_semantic_chunker/SETUP_ONCE.md](llm_semantic_chunker/SETUP_ONCE.md)

### "relation 'documents' does not exist" (n8n workflow)
- Your workflow is looking for the wrong table name
- Solution: Tables are named like `doc_filename_processed`
- Update your match_documents function to use the correct table name

### GitHub Security Alert - Exposed API Key

**If you see a GitHub security alert about an exposed API key:**

The key was removed from the current codebase but still exists in git history. To fully resolve:

1. **Revoke the exposed API key immediately:**
   - Go to [Google AI Studio](https://aistudio.google.com/app/apikey)
   - Delete the exposed API key
   - Generate a new API key

2. **Update your `.env` file:**
   - Replace the old key with the new key
   - Verify `.env` is in `.gitignore` (it is by default)

3. **GitHub will automatically close the alert** within 24-48 hours after detecting the key is revoked

**Prevention:**
- Never commit `.env` file or hardcode keys in code
- Always use environment variables for all credentials
- The `.gitignore` is already configured to exclude sensitive files

## Best Practices

1. **Security:**
   - Never hardcode API keys in code
   - Always use `.env` for credentials
   - Never commit `.env` to Git
   - Rotate API keys periodically

2. **API Usage:**
   - Use 4 Gemini API keys for rotation
   - Monitor your quota usage
   - Add `generate_embeddings=False` to chunking methods when testing

3. **Database Choice:**
   - **ChromaDB**: Simpler setup, managed cloud service
   - **Supabase**: More control, PostgreSQL access, hybrid search support

4. **Chunking:**
   - Use LLM semantic chunking for best results
   - Keep chunk sizes reasonable (400-800 tokens)
   - Filter out page headers and very short chunks

## Contributing

Contributions are welcome! Please follow these guidelines:

1. Fork the repository
2. Create a feature branch
3. Never commit API keys or `.env` files
4. Test your changes with both ChromaDB and Supabase
5. Submit a pull request

## License

[Add your license here]

## Support

For issues and questions:
- Check this README
- See `llm_semantic_chunker/SETUP_ONCE.md` for Supabase setup
- See `llm_semantic_chunker/README_SUPABASE.md` for Supabase details
- Open an issue on GitHub

## Acknowledgments

- **Google Gemini**: LLM and embedding model
- **ChromaDB**: Vector database
- **Supabase**: PostgreSQL + pgvector
- **LangChain**: Inspiration for semantic chunking approaches
