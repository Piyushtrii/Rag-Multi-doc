from typing import List, Optional

from pydantic import BaseModel, Field
"""Pydantic request/response schemas for the RAG API."""

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, description="The question or search query")
    k: Optional[int] = Field(5, ge=1, le=50, description="Number of documents to retrieve")
    use_reranking: Optional[bool] = Field(True, description="Whether to apply re-ranking")

    model_config = {
        "json_schema_extra": {
            "example": {
                "query": "What are the recent developments at Tesla?",
                "k": 5,
                "use_reranking": True,
            }
        }
    }


class RetrievalResult(BaseModel):
    id: str
    content: str
    source: str
    relevance_score: float


class QueryResponse(BaseModel):
    query: str
    answer: str
    sources: List[RetrievalResult]
    total_sources: int


class InitializeRequest(BaseModel):
    force_reload: Optional[bool] = Field(
        False,
        description="If true, clears the existing vector DB before loading documents",
    )


class RetrieveResponse(BaseModel):
    query: str
    results: List[RetrievalResult]
    total_results: int
