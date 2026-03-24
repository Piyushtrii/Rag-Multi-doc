# Enterprise RAG Chatbot

A production-grade Retrieval-Augmented Generation (RAG) API built with FastAPI, ChromaDB, and Groq.

**Features**
- **Multi-format ingestion** — TXT, PDF, DOCX, Markdown (recursive folder scan)
- **Hybrid search** — BM25 keyword + vector similarity, fused with configurable weights
- **Cross-encoder re-ranking** — scores candidate chunks against the query
- **Streaming responses** — token-level NDJSON stream via `/stream`
- **Layered package layout** — `core / services / pipeline / api`

---

### Add documents

Drop any supported files into `docs/`:

| Format | Extension |
|--------|-----------|
| Plain text | `.txt` |
| Markdown | `.md` |
| PDF | `.pdf` |
| Word document | `.docx` |

Sub-folders inside `docs/` are scanned automatically.

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Liveness / readiness probe |
| `GET` | `/stats` | Vector DB statistics |
| `POST` | `/initialize` | Load/reload documents into vector DB |
| `POST` | `/retrieve` | Retrieve relevant chunks (no generation) |
| `POST` | `/query` | Full RAG query (retrieve + generate) |
| `POST` | `/stream` | Streaming RAG response (NDJSON) |
---
