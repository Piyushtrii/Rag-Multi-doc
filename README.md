# Enterprise RAG Chatbot

A production-grade Retrieval-Augmented Generation (RAG) API built with FastAPI, ChromaDB, and Groq.

**Features**
- 📄 **Multi-format ingestion** — TXT, PDF, DOCX, Markdown (recursive folder scan)
- 🔍 **Hybrid search** — BM25 keyword + vector similarity, fused with configurable weights
- 🏆 **Cross-encoder re-ranking** — scores candidate chunks against the query
- ⚡ **Streaming responses** — token-level NDJSON stream via `/stream`
- 🧩 **Layered package layout** — `core / services / pipeline / api`

---

## Project Structure

```
CRAG/
├── app/
│   ├── core/           # Settings (config.py) and logging setup (logging.py)
│   ├── services/       # Embedding, LLM, VectorDB, Reranker adapters
│   ├── pipeline/       # DocumentLoader, HybridSearch, RAGPipeline
│   └── api/
│       ├── main.py     # FastAPI app factory (lifespan pattern)
│       ├── schemas.py  # Pydantic request/response models
│       └── routes/     # health.py  |  admin.py  |  query.py
├── docs/               # Drop your TXT / PDF / DOCX / MD files here
├── data/               # ChromaDB persisted data (git-ignored)
├── logs/               # Rotating log files (git-ignored)
├── evaluation/         # RAGAS evaluation scripts
├── scripts/
│   ├── init_db.py      # One-shot DB population script
│   └── run_server.py   # Dev server launcher
├── tests/              # pytest suite
├── .env.example        # Config template (copy → .env)
├── pyproject.toml
└── requirements.txt
```

---

## Quick Start

### 1. Clone & install

```bash
git clone <repo-url>
cd CRAG
python -m venv venv
# Windows:
.\venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Configure

```bash
cp .env.example .env
# Edit .env — set GROQ_API_KEY at minimum
```

### 3. Add documents

Drop any supported files into `docs/`:

| Format | Extension |
|--------|-----------|
| Plain text | `.txt` |
| Markdown | `.md` |
| PDF | `.pdf` |
| Word document | `.docx` |

Sub-folders inside `docs/` are scanned automatically.

### 4. Initialise the vector database

```bash
# First time
python -m scripts.init_db

# Force reload (wipes existing data)
python -m scripts.init_db --force
```

### 5. Start the server

```bash
# Development (auto-reload)
python -m scripts.run_server --reload

# Or directly with uvicorn
uvicorn app.api.main:app --reload --port 8000
```

Navigate to [http://localhost:8000/docs](http://localhost:8000/docs) for the interactive Swagger UI.

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Liveness / readiness probe |
| `GET` | `/stats` | Vector DB statistics |
| `POST` | `/initialize` | Load/reload documents into vector DB |
| `POST` | `/retrieve` | Retrieve relevant chunks (no generation) |
| `POST` | `/query` | Full RAG query (retrieve + generate) |
| `POST` | `/stream` | Streaming RAG response (NDJSON) |

See `API_DOCS.md` for full request/response examples.

---

## Running Tests

```bash
pytest tests/ -v
```

---

## Configuration Reference

All settings live in `.env` (see `.env.example`). Key options:

| Variable | Default | Description |
|----------|---------|-------------|
| `GROQ_API_KEY` | — | **Required** — Groq API key |
| `LLM_MODEL` | `openai/gpt-oss-120b` | Model served by Groq |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Sentence-Transformers model |
| `EMBEDDING_DEVICE` | `cpu` | `cpu` or `cuda` |
| `TOP_K_HYBRID` | `10` | Docs retrieved before re-ranking |
| `TOP_K_RERANK` | `5` | Docs returned after re-ranking |
| `BM25_WEIGHT` | `0.3` | BM25 weight in hybrid fusion |
| `VECTOR_WEIGHT` | `0.7` | Vector weight in hybrid fusion |
| `CHUNK_SIZE` | `500` | Max characters per chunk |
| `CHUNK_OVERLAP` | `50` | Overlap characters between chunks |
| `LOG_LEVEL` | `INFO` | `DEBUG` / `INFO` / `WARNING` / `ERROR` |
