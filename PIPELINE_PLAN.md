# Pipeline System Implementation Plan

## Overview

A production-grade pipeline system for document chunking and embedding with:
- Full orchestration (chunking → embedding → vector storage)
- Job management with SQLite (later PostgreSQL)
- Retry mechanisms and resume capability
- Comprehensive logging
- ZIP file support

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        pipeline.py (CLI)                        │
├─────────────────────────────────────────────────────────────────┤
│                      PipelineOrchestrator                       │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────┐   │
│  │ Chunker  │→ │ Embedder │→ │  Vector  │→ │   Reporter   │   │
│  │  Stage   │  │  Stage   │  │  Store   │  │              │   │
│  └──────────┘  └──────────┘  └──────────┘  └──────────────┘   │
├─────────────────────────────────────────────────────────────────┤
│                      JobManager (SQLite/PostgreSQL)             │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────────────┐   │
│  │  Jobs   │  │  Files  │  │ Chunks  │  │  Retry Queue    │   │
│  └─────────┘  └─────────┘  └─────────┘  └─────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

---

## Database Schema (SQLite)

### Table: `jobs`
```sql
CREATE TABLE jobs (
    id TEXT PRIMARY KEY,                    -- UUID
    name TEXT,                              -- Human-readable name
    status TEXT DEFAULT 'pending',          -- pending, running, paused, completed, failed
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    started_at TIMESTAMP,
    completed_at TIMESTAMP,

    -- Input config
    input_type TEXT,                        -- file, dir, zip
    input_path TEXT,
    recursive BOOLEAN DEFAULT TRUE,

    -- Chunker config
    chunker_type TEXT DEFAULT 'recursive',
    chunk_size INTEGER DEFAULT 512,
    chunk_overlap INTEGER DEFAULT 50,

    -- Embedder config
    embedding_provider TEXT,
    embedding_model TEXT,
    embedding_dimension INTEGER,

    -- Vector store config
    vector_store TEXT,
    collection_name TEXT,

    -- Progress
    total_files INTEGER DEFAULT 0,
    processed_files INTEGER DEFAULT 0,
    failed_files INTEGER DEFAULT 0,
    total_chunks INTEGER DEFAULT 0,
    total_embeddings INTEGER DEFAULT 0,

    -- Error tracking
    last_error TEXT,
    retry_count INTEGER DEFAULT 0,
    max_retries INTEGER DEFAULT 3
);
```

### Table: `job_files`
```sql
CREATE TABLE job_files (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id TEXT REFERENCES jobs(id),
    file_path TEXT,
    status TEXT DEFAULT 'pending',          -- pending, chunked, embedded, completed, failed

    -- Chunking results
    chunk_file_path TEXT,                   -- Path to _chunks.json
    num_chunks INTEGER,
    chunked_at TIMESTAMP,

    -- Embedding results
    embedded_at TIMESTAMP,

    -- Error tracking
    error_message TEXT,
    error_stage TEXT,                       -- chunk, embed, store
    retry_count INTEGER DEFAULT 0,
    last_attempt_at TIMESTAMP,

    UNIQUE(job_id, file_path)
);
```

### Table: `job_logs`
```sql
CREATE TABLE job_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id TEXT REFERENCES jobs(id),
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    level TEXT,                             -- DEBUG, INFO, WARNING, ERROR
    stage TEXT,                             -- init, chunk, embed, store
    message TEXT,
    file_path TEXT,
    details TEXT                            -- JSON for extra data
);
```

---

## CLI Interface

### New Entry Point: `pipeline.py`

```bash
# Start new job
python pipeline.py start \
    --input ./documents \
    --input-type dir \
    --chunker-type recursive \
    --chunk-size 512 \
    --embedding-provider ollama \
    --embedding-model mxbai-embed-large \
    --vector-store qdrant \
    --collection my_docs \
    --name "My Document Job"

# Start from ZIP
python pipeline.py start \
    --input ./docs.zip \
    --input-type zip \
    --chunker-type llm \
    --vector-store qdrant

# Resume paused/failed job
python pipeline.py resume <job_id>

# Retry failed files in a job
python pipeline.py retry <job_id>

# Pause running job
python pipeline.py pause <job_id>

# Check job status
python pipeline.py status <job_id>

# List all jobs
python pipeline.py list [--status pending|running|completed|failed]

# View job logs
python pipeline.py logs <job_id> [--level ERROR] [--tail 100]

# Delete job and its data
python pipeline.py delete <job_id> [--keep-vectors]

# Export job report
python pipeline.py report <job_id> --format json|html|csv
```

---

## Core Components

### 1. JobManager (`src/pipeline/job_manager.py`)

```python
class JobManager:
    def __init__(self, db_path: str = "pipeline.db"):
        """Initialize with SQLite connection"""

    def create_job(self, config: JobConfig) -> str:
        """Create new job, return job_id"""

    def get_job(self, job_id: str) -> Job:
        """Get job by ID"""

    def update_job_status(self, job_id: str, status: str):
        """Update job status"""

    def add_files(self, job_id: str, files: List[str]):
        """Add files to job"""

    def get_pending_files(self, job_id: str) -> List[JobFile]:
        """Get files pending processing"""

    def get_failed_files(self, job_id: str) -> List[JobFile]:
        """Get failed files for retry"""

    def mark_file_status(self, job_id: str, file_path: str, status: str, **kwargs):
        """Update file status with optional error info"""

    def log(self, job_id: str, level: str, stage: str, message: str, **kwargs):
        """Add log entry"""
```

### 2. PipelineOrchestrator (`src/pipeline/orchestrator.py`)

```python
class PipelineOrchestrator:
    def __init__(self, job_manager: JobManager, config: Config):
        """Initialize orchestrator"""

    def start_job(self, job_id: str):
        """Start or resume job processing"""

    def pause_job(self, job_id: str):
        """Gracefully pause job"""

    def process_file(self, job_id: str, file: JobFile) -> bool:
        """Process single file through pipeline"""

    def chunk_file(self, file_path: str, config: ChunkConfig) -> str:
        """Chunk file, return chunk file path"""

    def embed_chunks(self, chunk_file: str, config: EmbedConfig) -> List[List[float]]:
        """Generate embeddings for chunks"""

    def store_vectors(self, chunks: List, embeddings: List, config: StoreConfig):
        """Store in vector database"""

    def handle_error(self, job_id: str, file: JobFile, error: Exception, stage: str):
        """Handle and log errors, schedule retry if applicable"""
```

### 3. ZipHandler (`src/pipeline/zip_handler.py`)

```python
class ZipHandler:
    def __init__(self, temp_dir: str = None):
        """Initialize with temp directory"""

    def extract(self, zip_path: str) -> str:
        """Extract ZIP to temp dir, return extracted path"""

    def get_files(self, extracted_path: str, extensions: List[str]) -> List[str]:
        """Get list of processable files"""

    def cleanup(self, extracted_path: str):
        """Remove extracted files"""
```

### 4. PipelineLogger (`src/pipeline/logger.py`)

```python
class PipelineLogger:
    def __init__(self, job_manager: JobManager, job_id: str):
        """Initialize logger for job"""

    def debug(self, stage: str, message: str, **kwargs):
    def info(self, stage: str, message: str, **kwargs):
    def warning(self, stage: str, message: str, **kwargs):
    def error(self, stage: str, message: str, **kwargs):

    def get_logs(self, level: str = None, limit: int = 100) -> List[LogEntry]:
        """Retrieve logs with optional filtering"""
```

### 5. RetryManager (`src/pipeline/retry_manager.py`)

```python
class RetryManager:
    def __init__(self, job_manager: JobManager, max_retries: int = 3):
        """Initialize retry manager"""

    def should_retry(self, file: JobFile) -> bool:
        """Check if file should be retried"""

    def schedule_retry(self, job_id: str, file: JobFile):
        """Schedule file for retry with backoff"""

    def get_retry_delay(self, retry_count: int) -> int:
        """Calculate backoff delay: min(2^retry * 30, 600) seconds"""

    def process_retry_queue(self, job_id: str):
        """Process files due for retry"""
```

### 6. Reporter (`src/pipeline/reporter.py`)

```python
class Reporter:
    def __init__(self, job_manager: JobManager):
        """Initialize reporter"""

    def generate_report(self, job_id: str, format: str = 'json') -> str:
        """Generate report in specified format (json, html, csv)"""

    def get_summary(self, job_id: str) -> dict:
        """Get job summary statistics"""

    def get_failed_files_report(self, job_id: str) -> List[dict]:
        """Get detailed report of failed files"""
```

---

## Data Models (`src/pipeline/models.py`)

```python
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, List
from enum import Enum

class JobStatus(Enum):
    PENDING = 'pending'
    RUNNING = 'running'
    PAUSED = 'paused'
    COMPLETED = 'completed'
    FAILED = 'failed'

class FileStatus(Enum):
    PENDING = 'pending'
    CHUNKED = 'chunked'
    EMBEDDED = 'embedded'
    COMPLETED = 'completed'
    FAILED = 'failed'

@dataclass
class JobConfig:
    input_path: str
    input_type: str  # file, dir, zip
    chunker_type: str = 'recursive'
    chunk_size: int = 512
    chunk_overlap: int = 50
    embedding_provider: str = 'ollama'
    embedding_model: str = 'mxbai-embed-large'
    embedding_dimension: int = 1024
    vector_store: str = 'qdrant'
    collection_name: str = None
    name: str = None
    recursive: bool = True
    max_retries: int = 3

@dataclass
class Job:
    id: str
    name: str
    status: JobStatus
    config: JobConfig
    created_at: datetime
    started_at: Optional[datetime]
    completed_at: Optional[datetime]
    total_files: int
    processed_files: int
    failed_files: int
    total_chunks: int
    total_embeddings: int

@dataclass
class JobFile:
    id: int
    job_id: str
    file_path: str
    status: FileStatus
    chunk_file_path: Optional[str]
    num_chunks: Optional[int]
    chunked_at: Optional[datetime]
    embedded_at: Optional[datetime]
    error_message: Optional[str]
    error_stage: Optional[str]
    retry_count: int
    last_attempt_at: Optional[datetime]

@dataclass
class LogEntry:
    id: int
    job_id: str
    timestamp: datetime
    level: str
    stage: str
    message: str
    file_path: Optional[str]
    details: Optional[dict]
```

---

## File Structure

```
src/
├── pipeline/
│   ├── __init__.py
│   ├── models.py           # Data classes (Job, JobFile, JobConfig, etc.)
│   ├── job_manager.py      # Database operations
│   ├── orchestrator.py     # Main pipeline logic
│   ├── zip_handler.py      # ZIP extraction
│   ├── logger.py           # Logging to DB
│   ├── retry_manager.py    # Retry logic
│   └── reporter.py         # Report generation
├── chunkers/               # (existing)
├── embedders/              # (existing)
├── vector_stores/          # (existing)
└── utils/                  # (existing)

pipeline.py                 # CLI entry point
pipeline.db                 # SQLite database (auto-created)
```

---

## Implementation Phases

### Phase 1: Core Infrastructure
**Files to create:**
- `src/pipeline/__init__.py`
- `src/pipeline/models.py`
- `src/pipeline/job_manager.py`

**Tasks:**
1. Create data classes for Job, JobFile, JobConfig, LogEntry
2. Implement JobManager with SQLite operations
3. Create database schema with auto-migration
4. Add basic CRUD operations for jobs and files

### Phase 2: Pipeline Orchestration
**Files to create:**
- `src/pipeline/orchestrator.py`

**Tasks:**
1. Implement PipelineOrchestrator class
2. Integrate existing chunkers via `get_chunker()`
3. Integrate existing embedders via `get_embedder()`
4. Integrate vector stores via `get_vector_store()`
5. Implement file processing loop with status updates
6. Add graceful pause support (check flag between files)

### Phase 3: Error Handling & Retry
**Files to create:**
- `src/pipeline/retry_manager.py`

**Tasks:**
1. Implement RetryManager class
2. Add error classification (permanent vs temporary)
3. Implement exponential backoff calculation
4. Add retry queue with scheduled processing
5. Integrate with orchestrator error handling

### Phase 4: Logging & Monitoring
**Files to create:**
- `src/pipeline/logger.py`
- `src/pipeline/reporter.py`

**Tasks:**
1. Implement PipelineLogger with DB persistence
2. Add console progress display (tqdm or custom)
3. Implement Reporter with JSON/HTML/CSV output
4. Add job summary statistics

### Phase 5: ZIP Support
**Files to create:**
- `src/pipeline/zip_handler.py`

**Tasks:**
1. Implement ZipHandler class
2. Add temp directory management
3. Support nested directories in ZIP
4. Add cleanup on job completion or failure
5. Integrate with pipeline start command

### Phase 6: CLI Interface
**Files to create:**
- `pipeline.py`

**Tasks:**
1. Create CLI with argparse and subcommands
2. Implement `start` command with all options
3. Implement `resume` command
4. Implement `retry` command
5. Implement `pause` command
6. Implement `status` command
7. Implement `list` command with filtering
8. Implement `logs` command with level filter
9. Implement `delete` command
10. Implement `report` command
11. Add help text and usage examples

### Phase 7: Testing & Polish
**Tasks:**
1. Unit tests for JobManager
2. Unit tests for RetryManager
3. Integration tests for full pipeline
4. Error handling edge cases
5. Documentation and examples

---

## Configuration (.env additions)

```bash
# Pipeline settings
PIPELINE_DB_PATH=pipeline.db
PIPELINE_MAX_RETRIES=3
PIPELINE_RETRY_DELAY=30
PIPELINE_TEMP_DIR=./temp
PIPELINE_LOG_LEVEL=INFO

# Future PostgreSQL support
# PIPELINE_DB_TYPE=postgresql
# PIPELINE_DB_URL=postgresql://user:pass@host:5432/dbname
```

---

## Usage Examples

### Basic Pipeline Run
```bash
python pipeline.py start \
    --input ./alibaba-docs-cleaned \
    --chunker-type recursive \
    --chunk-size 400 \
    --chunk-overlap 50 \
    --embedding-provider ollama \
    --embedding-model mxbai-embed-large \
    --vector-store qdrant \
    --collection alibaba_docs \
    --name "Alibaba Docs Pipeline"
```

**Expected Output:**
```
Job created: abc123-def456
Scanning input directory...
Found 3037 files to process

Processing files...
[========================================] 100% | 3037/3037 | 76913 chunks | ETA: 0:00

Job completed: abc123-def456
  Duration:     2h 15m 32s
  Total files:  3037
  Successful:   3000
  Failed:       37
  Total chunks: 76913
  Embeddings:   76913

Run 'python pipeline.py retry abc123-def456' to retry failed files.
```

### Process from ZIP
```bash
python pipeline.py start \
    --input ./documents.zip \
    --input-type zip \
    --chunker-type llm \
    --vector-store qdrant \
    --collection uploaded_docs
```

### Resume Interrupted Job
```bash
python pipeline.py resume abc123-def456
```

### Retry Failed Files
```bash
python pipeline.py retry abc123-def456
```

### Check Status
```bash
python pipeline.py status abc123-def456
```

**Expected Output:**
```
Job: abc123-def456
Name: Alibaba Docs Pipeline
Status: completed

Progress:
  Files:      3037/3037 (100%)
  Successful: 3000
  Failed:     37
  Chunks:     76913
  Embeddings: 76913

Timing:
  Created:   2024-12-09 15:30:00
  Started:   2024-12-09 15:30:05
  Completed: 2024-12-09 17:45:37
  Duration:  2h 15m 32s

Configuration:
  Input:      ./alibaba-docs-cleaned
  Chunker:    recursive (400 tokens, 50 overlap)
  Embedder:   ollama/mxbai-embed-large (1024 dims)
  VectorDB:   qdrant/alibaba_docs
```

### List Jobs
```bash
python pipeline.py list --status completed
```

### View Logs
```bash
python pipeline.py logs abc123-def456 --level ERROR --tail 50
```

### Generate Report
```bash
python pipeline.py report abc123-def456 --format html > report.html
```

---

## Future Enhancements (Not in Initial Scope)

1. **PostgreSQL Support** - Swap SQLite for PostgreSQL for production
2. **Distributed Processing** - Multiple workers with job queue (Redis/RabbitMQ)
3. **Web Dashboard** - Real-time monitoring UI with progress bars
4. **Webhooks** - Notifications on job completion/failure
5. **Scheduling** - Cron-like scheduled pipeline runs
6. **Incremental Updates** - Detect changed files and re-process only those
7. **S3/Cloud Storage** - Input from cloud storage buckets
8. **API Server** - REST API for job management
9. **Metrics Export** - Prometheus/Grafana integration

---

## Dependencies

No new dependencies required. Uses existing:
- `sqlite3` (Python standard library)
- `argparse` (Python standard library)
- `zipfile` (Python standard library)
- `tempfile` (Python standard library)
- `uuid` (Python standard library)
- `tqdm` (already in project for progress bars)

---

## Notes for Implementation

1. **Reuse existing code**: The chunkers, embedders, and vector stores are already well-designed. The pipeline wraps them without modification.

2. **Graceful pause**: Check a `should_stop` flag between file processing iterations. Set via signal handler (SIGINT) or pause command.

3. **Atomic file status updates**: Use SQLite transactions to ensure file status is always consistent.

4. **Progress persistence**: Job state is saved to DB after each file, so crashes lose at most one file's work.

5. **Error classification**:
   - Permanent: Invalid file format, empty file, parse error
   - Temporary: Network timeout, rate limit, service unavailable

6. **Backward compatibility**: Keep existing `chunker.py` and `embedder.py` working as-is. Pipeline is an additional tool, not a replacement.
