# Sample Document for Enterprise RAG

## Introduction
This is a sample document to test the Enterprise RAG Chatbot ingestion pipeline.
It covers various topics that can be retrieved later.

## ChromaDB Storage
ChromaDB is used as the vector database to store document embeddings. When you run the `init_db.py` script, it reads files from the `docs/` directory, splits them into chunks, creates embeddings for each chunk using SentenceTransformers, and stores them in a local ChromaDB instance.

## Retrieval Process
When a user asks a question, the query is also embedded. The system performs a hybrid search combining BM25 keyword matching and vector similarity search. The cross-encoder reranker then scores the retrieved chunks to provide the best context to the LLM.
