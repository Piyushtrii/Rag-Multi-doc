from typing import AsyncGenerator, Dict, List

import mlflow
from app.core.mlflow_config import init_mlflow

from loguru import logger

from app.core.config import settings
from app.pipeline.document_loader import DocumentLoader
from app.pipeline.hybrid_search import HybridSearchEngine
from app.services.embedding import EmbeddingService
from app.services.llm import LLMService
from app.services.reranker import Reranker
from app.services.vector_db import VectorDBManager

"""RAG pipeline — orchestrates retrieval and generation end-to-end."""
class RAGPipeline:
    """Complete RAG pipeline: load → embed → retrieve → re-rank → generate."""

    def __init__(self, use_reranking: bool = True, use_hybrid_search: bool = True,) -> None:
        logger.info("Initialising RAG Pipeline")

        self.embedding_service = EmbeddingService()
        self.vector_db = VectorDBManager(self.embedding_service)
        self.document_loader = DocumentLoader()

        self.use_reranking = use_reranking
        self.use_hybrid_search = use_hybrid_search
        self.hybrid_search: HybridSearchEngine | None = None
        self.reranker: Reranker | None = Reranker() if use_reranking else None
        self.llm_service = LLMService()

        logger.info("RAG Pipeline initialised")

    #Initialisation

    def initialize_database(self) -> None:
        """Load documents from disk, chunk them, and populate the vector store."""
        logger.info("Initialising database with documents")

        raw_docs = self.document_loader.load_documents()
        if not raw_docs:
            logger.warning("No documents loaded — vector DB will be empty")
            return

        chunked_docs = self.document_loader.chunk_documents(raw_docs)
        self.vector_db.add_documents(chunked_docs)

        if self.use_hybrid_search:
            self.hybrid_search = HybridSearchEngine(chunked_docs)

        logger.info("Database initialised successfully")

    # Retrieval 

    def retrieve(self, query: str, k: int | None = None) -> List[Dict]:
        """Return the top-*k* most relevant document chunks for *query*."""
        k = k or settings.TOP_K_HYBRID
        logger.info(f"Retrieving documents for query: '{query}'")

        vector_results = self.vector_db.search(query, k=k)

        if self.use_hybrid_search and self.hybrid_search:
            results = self.hybrid_search.hybrid_search(query, vector_results, k=k)
        else:
            results = vector_results

        if self.use_reranking and self.reranker:
            results = self.reranker.rerank(query, results, k=settings.TOP_K_RERANK)

        logger.info(f"Retrieved {len(results)} documents")
        return results

    #Generation

    def generate(self, query: str, context: str) -> str:
        """Generate a complete answer for *query* given *context*."""
        logger.info(f"Generating response for query: '{query}'")
        return self.llm_service.call_llm(query, context)

    async def generate_streaming(self, query: str, context: str) -> AsyncGenerator[str, None]:
        """Yield streamed LLM tokens for *query* / *context*."""
        logger.info(f"Streaming response for query: '{query}'")
        async for chunk in self.llm_service.call_llm_streaming_async(query, context):
            yield chunk

    #Combined query 

    '''def query(self, query: str, k: int | None = None) -> Dict:
        """Retrieve + generate in one call. Returns answer + source metadata."""
        retrieved_docs = self.retrieve(query, k=k)
        context = self._prepare_context(retrieved_docs)
        answer = self.generate(query, context)

        return {
            "query": query,
            "answer": answer,
            "sources": [
                {
                    "id": doc["id"],
                    "source": doc["metadata"].get("source", "Unknown"),
                    "relevance_score": doc.get(
                        "rerank_score", doc.get("combined_score", 0.0)
                    ),
                }
                for doc in retrieved_docs
            ],
        }

    async def query_streaming(
        self, query: str, k: int | None = None
    ) -> AsyncGenerator[str, None]:
        """Retrieve + stream LLM response."""
        retrieved_docs = self.retrieve(query, k=k)
        context = self._prepare_context(retrieved_docs)
        async for chunk in self.generate_streaming(query, context):
            yield chunk'''
    

    def query(self, query: str, k: int | None = None) -> Dict:
   
        init_mlflow()

        with mlflow.start_run(run_name="rag_query"):

            mlflow.log_param("query", query)
            mlflow.log_param("top_k", k or settings.TOP_K_HYBRID)
            mlflow.log_param("use_reranking", self.use_reranking)
            mlflow.log_param("use_hybrid_search", self.use_hybrid_search)

            #Retrieval
            retrieved_docs = self.retrieve(query, k=k)
            mlflow.log_metric("num_docs_retrieved", len(retrieved_docs))

            #Context
            context = self._prepare_context(retrieved_docs)

            #Generation
            answer = self.generate(query, context)

            #Logs
            mlflow.log_text(context, "context.txt")
            mlflow.log_text(answer, "answer.txt")

            #Document scores
            for i, doc in enumerate(retrieved_docs):
                score = doc.get("rerank_score", doc.get("combined_score", 0.0))
                mlflow.log_metric(f"doc_{i}_score", score)

            return {
                "query": query,
                "answer": answer,
                "sources": [
                    {
                        "id": doc["id"],
                        "source": doc["metadata"].get("source", "Unknown"),
                        "relevance_score": doc.get(
                            "rerank_score", doc.get("combined_score", 0.0)
                        ),
                    }
                    for doc in retrieved_docs
                ],
            }
        
    async def query_streaming(self, query: str, k: int | None = None) -> AsyncGenerator[str, None]:
        init_mlflow()
        with mlflow.start_run(run_name="rag_stream_query"):
            mlflow.log_param("query", query)

            retrieved_docs = self.retrieve(query, k=k)
            context = self._prepare_context(retrieved_docs)

            mlflow.log_metric("num_docs_retrieved", len(retrieved_docs))
            mlflow.log_text(context, "context.txt")

            async for chunk in self.generate_streaming(query, context):
                yield chunk
        
    #Utilities 
    def _prepare_context(self, documents: List[Dict]) -> str:
        """Concatenate document content into a single context string."""
        parts = [
            f"Source: {doc['metadata'].get('source', 'Unknown')}\n\n{doc['content']}"
            for doc in documents
        ]
        return "\n\n---\n\n".join(parts)

    def get_stats(self) -> Dict:
        """Return pipeline statistics."""
        return {
            "vector_db": self.vector_db.get_stats(),
            "use_reranking": self.use_reranking,
            "use_hybrid_search": self.use_hybrid_search,
        }

