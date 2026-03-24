from typing import Dict, List

import chromadb
from loguru import logger

from app.core.config import settings
from app.services.embedding import EmbeddingService
"""Vector database manager — wraps ChromaDB for persistent document storage."""


class VectorDBManager:
    """Manages a ChromaDB persistent collection for document embeddings."""

    def __init__(self, embedding_service: EmbeddingService | None = None) -> None:
        self.embedding_service = embedding_service or EmbeddingService()
        self.db_path = settings.DB_PATH
        self.db_path.mkdir(parents=True, exist_ok=True)

        logger.info(f"Initialising ChromaDB at {self.db_path}")
        self.client = chromadb.PersistentClient(path=str(self.db_path))
        self.collection = self.client.get_or_create_collection(
            name=settings.VECTOR_DB_COLLECTION,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(f"Collection '{settings.VECTOR_DB_COLLECTION}' ready")

    #Write
    def add_documents(self, documents: List[Dict]) -> None:
        """Embed *documents* and upsert them into the collection."""
        logger.info(f"Adding {len(documents)} documents to vector DB")
        ids = [doc["id"] for doc in documents]
        contents = [doc["content"] for doc in documents]
        metadatas = [doc["metadata"] for doc in documents]

        embeddings = self.embedding_service.embed_texts(contents)
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=contents,
            metadatas=metadatas,
        )
        logger.info("Successfully added documents to vector DB")

    def delete_all(self) -> None:
        """Drop and recreate the collection (effectively clearing all data)."""
        logger.warning("Deleting all documents from vector DB")
        self.client.delete_collection(name=settings.VECTOR_DB_COLLECTION)
        self.collection = self.client.get_or_create_collection(
            name=settings.VECTOR_DB_COLLECTION,
            metadata={"hnsw:space": "cosine"},
        )
        # NOTE: PersistentClient auto-persists; no explicit persist() call needed.
        logger.info("Vector DB cleared")

    #Read

    def search(self, query: str, k: int = 10) -> List[Dict]:
        """Return the top-*k* documents most similar to *query*."""
        logger.info(f"Vector search for query: '{query}'")
        query_embedding = self.embedding_service.embed_query(query)

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(k, self.collection.count()) or 1,
        )

        documents: List[Dict] = []
        if results and results["documents"]:
            for i, doc_id in enumerate(results["ids"][0]):
                distance = results["distances"][0][i] if results.get("distances") else 0
                documents.append(
                    {
                        "id": doc_id,
                        "content": results["documents"][0][i],
                        "metadata": results["metadatas"][0][i] if results.get("metadatas") else {},
                        "distance": distance,
                        "score": 1 - distance,
                    }
                )
        return documents

    def get_all_documents(self) -> List[Dict]:
        """Retrieve every document stored in the collection."""
        logger.info("Retrieving all documents from vector DB")
        results = self.collection.get()
        documents: List[Dict] = []
        if results and results["documents"]:
            for i, doc_id in enumerate(results["ids"]):
                documents.append(
                    {
                        "id": doc_id,
                        "content": results["documents"][i],
                        "metadata": results["metadatas"][i] if results.get("metadatas") else {},
                    }
                )
        return documents

    def get_stats(self) -> Dict:
        """Return basic collection statistics."""
        return {
            "collection_name": settings.VECTOR_DB_COLLECTION,
            "document_count": self.collection.count(),
        }
