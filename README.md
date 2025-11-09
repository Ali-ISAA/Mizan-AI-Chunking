# MizanAiChunking

Advanced document chunking system with LLM-based semantic analysis and RAG chatbot capabilities.

## Overview

This project provides multiple intelligent document chunking methods, with a focus on LLM-based semantic chunking using Google Gemini. It includes a complete RAG (Retrieval-Augmented Generation) chatbot system powered by ChromaDB.

## Project Structure

```
MizanAiChunking/
├── llm_semantic_chunker/     # LLM-based semantic chunking (PRIMARY)
├── other_chunkers/            # Alternative chunking methods
├── chatbot/                   # RAG chatbot system
├── MD_FILES/                  # Place your .md files here for processing
├── docs/                      # Documentation
├── tests/                     # Test files
├── utils/                     # Utility scripts
└── output/                    # Output files (git-ignored)
```

## Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <repository-url>
cd MizanAiChunking

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration

```bash
# Copy the environment template
cp .env.example .env

# Edit .env and add your API keys
# - GEMINI_API_KEY_1, GEMINI_API_KEY_2, etc.
# - CHROMADB_API_KEY, CHROMADB_TENANT, CHROMADB_DATABASE
```

### 3. Add Your Markdown Files

```bash
# Place your .md files in the MD_FILES folder
# The chunker will automatically detect and list them
cp your_document.md MD_FILES/
```

### 4. Run LLM Semantic Chunker

```bash
cd llm_semantic_chunker
python llm_semantic_md_chunker.py

# The script will:
# 1. List all .md files in MD_FILES/ folder
# 2. Let you select which file to process
# 3. Perform LLM semantic analysis
# 4. Save chunks to ChromaDB for RAG retrieval
```

### 5. Run Chatbot

```bash
cd chatbot
python chatbot.py

# The chatbot will:
# 1. List all available collections in ChromaDB
# 2. Let you select which collection to query
# 3. Start interactive Q&A mode
```

## Modules

### 📁 llm_semantic_chunker/
**LLM-Based Semantic Chunking** - Uses Google Gemini to intelligently chunk documents

- `llm_semantic_md_chunker.py` - Markdown semantic chunker
- `llm_semantic_pdf_chunker.py` - PDF semantic chunker
- `api_key_manager.py` - API key rotation manager
- `file_reader.py` - File reading utilities

**Features:**
- Pure semantic analysis (no size limits)
- Automatic API key rotation
- ChromaDB integration
- Post-processing for oversized chunks

### 📁 other_chunkers/
**Alternative Chunking Methods**

- `context_aware_md_chunker.py` - Context-aware markdown chunking
- `context_aware_pdf_chunker.py` - Context-aware PDF chunking
- `markdown_section_chunker.py` - Section-based markdown chunking
- `text_chunking_methods.py` - Traditional chunking methods

### 📁 chatbot/
**RAG Chatbot System**

- `chatbot.py` - Main chatbot with ChromaDB + Gemini
- `chatbot_ui.py` - Streamlit web interface
- `simple_chatbot.py` - Simplified version

**Features:**
- Retrieval-Augmented Generation (RAG)
- ChromaDB vector search
- Conversation history
- Multi-language support (English/Arabic)

### 📁 docs/
Complete documentation including setup guides, API rate limits, and troubleshooting

### 📁 tests/
Test scripts for various components

### 📁 utils/
Utility scripts including example usage and ChromaDB management

## Environment Variables

Required environment variables (see `.env.example`):

```bash
# Google Gemini API Keys (for rotation)
GEMINI_API_KEY_1=your_key_here
GEMINI_API_KEY_2=your_key_here
GEMINI_API_KEY_3=your_key_here
GEMINI_API_KEY_4=your_key_here

# ChromaDB Cloud
CHROMADB_API_KEY=your_chromadb_key
CHROMADB_TENANT=your_tenant_id
CHROMADB_DATABASE=your_database_name
```

## Key Features

- **LLM Semantic Chunking**: Uses Gemini AI to understand content and create meaningful chunks
- **Automatic Size Management**: Post-processes oversized chunks while maintaining semantic boundaries
- **API Key Rotation**: Automatic failover between multiple API keys
- **Multi-format Support**: Markdown, PDF (with OCR), and text files
- **ChromaDB Integration**: Vector database for semantic search
- **RAG Chatbot**: Complete chatbot system with conversation memory

## Documentation

See the `docs/` folder for detailed documentation:
- `SETUP_GUIDE.md` - Initial setup instructions
- `QUICK_START.md` - Quick start guide
- `API_RATE_LIMITS.md` - API usage limits and optimization
- `CHATBOT_QUICKSTART.md` - Chatbot usage guide
- And more...

## License

[Add your license here]

## Contributing

[Add contribution guidelines here]

## Support

For issues and questions, please check the documentation in the `docs/` folder.
