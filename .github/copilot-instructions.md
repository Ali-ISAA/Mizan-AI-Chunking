<!-- Copilot / AI agent instructions for working in this repo -->
# Mizan-AI-Chunking — Copilot Instructions

Purpose: Give concise, actionable context so an AI coding agent can be immediately productive.

- **Entry points:** `chunker.py` (document splitting) and `embedder.py` (embed + store).
- **Core directories:** `src/chunkers/`, `src/embedders/`, `src/llms/`, `src/vector_stores/`, `src/utils/`.

Big picture (quick):
- This is a modular pipeline: Chunkers -> (optional) LLMs -> Embedders -> VectorStores.
- Plugins use a registry pattern: each plugin is registered in the package `__init__.py` (e.g. `src/chunkers/__init__.py`).
- Base classes define the contract: `BaseChunker`, `BaseEmbedder`, `BaseLLM`, `BaseVectorStore` under respective folders.

What to know before editing:
- Config and secrets live in `.env` (use `.env.example` to bootstrap). Do not hardcode keys.
- `src/utils/config.py` exposes typed config; use it rather than reading `.env` directly.
- `src/utils/api_key_manager.py` implements multi-key rotation (Gemini). Use its `execute_with_retry` helpers for provider calls.
- Text ingestion always uses `get_file_text()` in `src/utils/file_reader.py` — prefer it for consistent parsing across formats.

Developer workflows & common commands:
- Install: `pip install -r requirements.txt`.
- Bootstrap env: `cp .env.example .env` and edit keys.
- Run chunker (default recommended): `python chunker.py --file document.md`.
- Run embedder (all-in-one): `python embedder.py --file document.md`.
- See examples: `bash examples.sh` and `python chunker.py --help` / `python embedder.py --help`.

Project-specific conventions (do not assume generic defaults):
- Registry pattern: new plugins must be added to the module `__init__` mapping (e.g. add `'my': MyClass` in `src/chunkers/__init__.py`).
- Chunk API: chunkers return dicts with keys `text`, `tokens`, and `metadata` (include `source_file` and `chunk_index`). Follow this shape exactly.
- Base classes: implement required methods (`chunk()`, `generate()`, `embed()`, `insert()`/`search()` etc.) to be loadable by registries.
- Token limits: semantic chunkers enforce max token sizes (e.g. 1024); preserve those caps when changing chunk logic.

Integration points & dependencies to watch:
- Embedding providers (Gemini, OpenAI, Ollama) are under `src/embedders/` — embedding calls are rate-limited and may use the API key manager.
- Vector stores (ChromaDB, Supabase/pgvector, Qdrant, Weaviate, Pinecone) live in `src/vector_stores/` — each implements `create_collection`, `insert`, and `search`.
- LLM providers used by the `llm` chunker are under `src/llms/` — prefer using the `litellm` abstraction when available.

How to add a new plugin (short checklist):
1. Create implementation file under the appropriate folder (e.g. `src/chunkers/my_chunker.py`).
2. Inherit from the corresponding `Base*` class and implement required methods.
3. Register the class in the package `__init__.py` mapping so CLIs can find it.
4. Add any `.env` keys needed to `.env.example` and update `SETUP_GUIDE.md` if provider setup is nontrivial.

Debugging tips specific to this repo:
- To test chunkers without spending embedding quota: run `python chunker.py --file test.md --output test_chunks.json` and inspect the JSON.
- For provider quota issues (Gemini/OpenAI), add more keys to `.env` (e.g. `GEMINI_API_KEY_2`) — the `APIKeyManager` rotates automatically.
- If a vector store connection fails, check `.env` credentials and service status; Supabase/pgvector expect the pgvector extension enabled.

Files with further detailed guidance: `CLAUDE.md`, `SETUP_GUIDE.md`, and the repo `README.md`.

If you want me to include code-snippets or wire up CI checks (lint/tests) next, tell me which to prioritize.
