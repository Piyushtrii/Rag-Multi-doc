from typing import Dict, List, Tuple

from loguru import logger
from sentence_transformers import CrossEncoder

from app.core.config import settings
"""Re-ranker service — scores document-query pairs with a cross-encoder model."""

class Reranker:
    """Re-rank search results using a cross-encoder relevance model."""

    def __init__(self, model_name: str | None = None) -> None:
        self.model_name = model_name or settings.RERANKER_MODEL
        logger.info(f"Loading reranker model: {self.model_name}")
        self.model = CrossEncoder(self.model_name)
        logger.info("Reranker model loaded successfully")

    def rerank(
        self,
        query: str,
        documents: List[Dict],
        k: int | None = None,
        threshold: float | None = None,
    ) -> List[Dict]:
        """Re-rank *documents* by relevance to *query* and return the top-*k*."""
        k = k or settings.TOP_K_RERANK
        threshold = threshold if threshold is not None else settings.RERANKER_THRESHOLD

        if not documents:
            return []

        logger.info(f"Re-ranking {len(documents)} documents for query: '{query}'")

        pairs = [[query, doc["content"]] for doc in documents]
        scores = self.model.predict(pairs)

        for i, doc in enumerate(documents):
            doc["rerank_score"] = float(scores[i])

        if threshold > 0:
            documents = [d for d in documents if d["rerank_score"] >= threshold]

        reranked = sorted(documents, key=lambda x: x["rerank_score"], reverse=True)
        return reranked[:k]

    def rerank_and_score(
        self,
        query: str,
        documents: List[Dict],
    ) -> List[Tuple[Dict, float]]:
        """Re-rank and return ``(document, score)`` tuples."""
        reranked = self.rerank(query, documents)
        return [(doc, doc["rerank_score"]) for doc in reranked]
