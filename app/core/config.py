"""Application settings — single source of truth for all configuration."""

from pathlib import Path
from typing import Optional

from pydantic import ConfigDict
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    #Project paths
    PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]  # CRAG/
    DOCS_PATH: Path = PROJECT_ROOT / "docs"
    DB_PATH: Path = PROJECT_ROOT / "data" / "vector_db"
    LOG_PATH: Path = PROJECT_ROOT / "logs"

    # LLM
    LLM_PROVIDER: str = "groq"
    LLM_MODEL: str = "openai/gpt-oss-120b"
    LLM_TEMPERATURE: float = 0.7
    LLM_MAX_TOKENS: int = 1024
    GROQ_API_KEY: Optional[str] = None
    HUGGINGFACEHUB_API_TOKEN: Optional[str] = None

    #Embedding
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    EMBEDDING_DEVICE: str = "cpu"  # "cpu" or "cuda"

    #Vector DB
    VECTOR_DB_TYPE: str = "chromadb"
    VECTOR_DB_COLLECTION: str = "enterprise_docs"

    #Search
    TOP_K_HYBRID: int = 10
    TOP_K_RERANK: int = 5
    BM25_WEIGHT: float = 0.3
    VECTOR_WEIGHT: float = 0.7

    #Re-ranker
    RERANKER_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    RERANKER_THRESHOLD: float = 0.0

    #API
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    API_WORKERS: int = 1

    #Logging
    LOG_LEVEL: str = "INFO"

    #Document processing
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 50

    model_config = ConfigDict(
        # Resolve .env relative to project root so it's found regardless of CWD
        env_file=str(Path(__file__).resolve().parents[2] / ".env"),
        env_file_encoding="utf-8",
        case_sensitive=True,
    )


settings = Settings()
