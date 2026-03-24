from typing import List

import numpy as np
from loguru import logger
from sentence_transformers import SentenceTransformer

from app.core.config import settings

"""Embedding service — generates vector representations for text."""
class EmbeddingService:
    """Generate embeddings using a sentence-transformers model."""

    def __init__(self, model_name: str | None = None) -> None:
        self.model_name = model_name or settings.EMBEDDING_MODEL
        logger.info(f"Loading embedding model: {self.model_name}")

        self.model = SentenceTransformer(
            self.model_name,
            device=settings.EMBEDDING_DEVICE,
        )
        self.embedding_dim: int = self.model.get_sentence_embedding_dimension()
        logger.info(f"Embedding dimension: {self.embedding_dim}")

    #Public API

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Return embeddings for *texts* as a list of float lists."""
        try:
            embeddings = self.model.encode(
                texts,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
            return embeddings.tolist() if isinstance(embeddings, np.ndarray) else embeddings
        except Exception as exc:
            logger.error(f"Error generating embeddings: {exc}")
            raise

    def embed_text(self, text: str) -> List[float]:
        """Return embedding for a single *text*."""
        return self.embed_texts([text])[0]

    def embed_query(self, query: str) -> List[float]:
        """Embed a user *query* (can be specialised differently from documents)."""
        return self.embed_text(query)
