from typing import Dict, List

import numpy as np
from loguru import logger
from rank_bm25 import BM25Okapi

from app.core.config import settings
"""Hybrid search engine — combines BM25 keyword search with vector similarity scores."""

class HybridSearchEngine:
    """Fuse BM25 and vector search results using weighted score combination."""

    def __init__(self, documents: List[Dict] | None = None) -> None:
        self.documents: List[Dict] = documents or []
        self.bm25_model: BM25Okapi | None = None
        self.tokenized_docs: List[List[str]] | None = None

        if documents:
            self._build_bm25_index(documents)

    #Index

    def _build_bm25_index(self, documents: List[Dict]) -> None:
        logger.info(f"Building BM25 index for {len(documents)} documents")
        self.documents = documents
        self.tokenized_docs = [doc["content"].lower().split() for doc in documents]
        self.bm25_model = BM25Okapi(self.tokenized_docs)
        logger.info("BM25 index built successfully")

    #Search

    def bm25_search(self, query: str, k: int = 10) -> List[Dict]:
        """Return top-*k* results from BM25 keyword matching."""
        if not self.bm25_model:
            logger.warning("BM25 model not initialised")
            return []

        scores = self.bm25_model.get_scores(query.lower().split())
        top_indices = np.argsort(scores)[-k:][::-1]

        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                results.append(
                    {
                        "id": self.documents[idx]["id"],
                        "content": self.documents[idx]["content"],
                        "metadata": self.documents[idx]["metadata"],
                        "score": float(scores[idx]),
                        "search_type": "bm25",
                    }
                )
        return results

    def hybrid_search(
        self,
        query: str,
        vector_results: List[Dict],
        k: int = 10,
        bm25_weight: float | None = None,
        vector_weight: float | None = None,
    ) -> List[Dict]:
        """Combine BM25 + vector results into a single ranked list."""
        bm25_weight = bm25_weight or settings.BM25_WEIGHT
        vector_weight = vector_weight or settings.VECTOR_WEIGHT

        logger.info(
            f"Hybrid search: bm25_w={bm25_weight}, vec_w={vector_weight}, k={k}"
        )

        bm25_results = self.bm25_search(query, k=k)
        bm25_results, vector_results = self._normalize_scores(bm25_results, vector_results)

        combined: Dict[str, Dict] = {}

        for r in bm25_results:
            combined[r["id"]] = {
                "id": r["id"],
                "content": r["content"],
                "metadata": r["metadata"],
                "bm25_score": r["score"],
                "vector_score": 0.0,
            }

        for r in vector_results:
            doc_id = r["id"]
            if doc_id in combined:
                combined[doc_id]["vector_score"] = r["score"]
            else:
                combined[doc_id] = {
                    "id": doc_id,
                    "content": r["content"],
                    "metadata": r["metadata"],
                    "bm25_score": 0.0,
                    "vector_score": r["score"],
                }

        for doc in combined.values():
            doc["combined_score"] = (
                doc["bm25_score"] * bm25_weight + doc["vector_score"] * vector_weight
            )

        return sorted(combined.values(), key=lambda x: x["combined_score"], reverse=True)[:k]

    #Helpers

    @staticmethod
    def _normalize_scores(
        bm25_results: List[Dict],
        vector_results: List[Dict],
    ) -> tuple[List[Dict], List[Dict]]:
        """Normalise BM25 scores to [0, 1]; ensure vector results have a 'score' key."""
        if bm25_results:
            max_bm25 = max(r["score"] for r in bm25_results) or 1.0
            for r in bm25_results:
                r["score"] = r["score"] / max_bm25

        for r in vector_results:
            if "score" not in r:
                r["score"] = 1.0 - r.get("distance", 0.0)

        return bm25_results, vector_results
