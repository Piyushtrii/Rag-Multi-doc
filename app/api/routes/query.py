import json

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from loguru import logger

from app.api.schemas import (
    QueryRequest,
    QueryResponse,
    RetrievalResult,
    RetrieveResponse,
)
"""Query routes — retrieve, non-streaming query, and streaming query."""

router = APIRouter(tags=["Query"])


@router.post("/retrieve", response_model=RetrieveResponse)
async def retrieve(request: Request, body: QueryRequest):
    """Retrieve relevant document chunks without generating an answer."""
    pipeline = request.app.state.rag_pipeline
    if not body.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    try:
        logger.info(f"Retrieving documents for query: '{body.query}'")
        documents = pipeline.retrieve(body.query, k=body.k)
        results = [
            RetrievalResult(
                id=doc["id"],
                content=doc["content"][:500] + "…" if len(doc["content"]) > 500 else doc["content"],
                source=doc["metadata"].get("source", "Unknown"),
                relevance_score=doc.get("rerank_score", doc.get("combined_score", 0.0)),
            )
            for doc in documents
        ]
        return RetrieveResponse(
            query=body.query,
            results=results,
            total_results=len(results),
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"Retrieval failed: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/query", response_model=QueryResponse)
async def query(request: Request, body: QueryRequest):
    """Complete RAG query — retrieve docs and generate an answer."""
    pipeline = request.app.state.rag_pipeline
    if not body.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    try:
        logger.info(f"Processing query: '{body.query}'")
        result = pipeline.query(body.query, k=body.k)
        sources = [
            RetrievalResult(
                id=src["id"],
                content="",
                source=src["source"],
                relevance_score=src["relevance_score"],
            )
            for src in result["sources"]
        ]
        return QueryResponse(
            query=result["query"],
            answer=result["answer"],
            sources=sources,
            total_sources=len(sources),
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"Query processing failed: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/stream")
async def stream_query(request: Request, body: QueryRequest):
    """Stream the RAG response token-by-token (NDJSON format)."""
    pipeline = request.app.state.rag_pipeline
    if not body.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    async def _generator():
        try:
            documents = pipeline.retrieve(body.query, k=body.k)
            context = pipeline._prepare_context(documents)

            metadata = {
                "query": body.query,
                "sources": [
                    {
                        "id": doc["id"],
                        "source": doc["metadata"].get("source", "Unknown"),
                        "relevance_score": doc.get(
                            "rerank_score", doc.get("combined_score", 0.0)
                        ),
                    }
                    for doc in documents
                ],
            }
            yield json.dumps({"type": "metadata", "data": metadata}) + "\n"

            async for chunk in pipeline.generate_streaming(body.query, context):
                yield json.dumps({"type": "content", "data": chunk}) + "\n"

            yield json.dumps({"type": "done"}) + "\n"
        except Exception as exc:
            logger.error(f"Streaming error: {exc}")
            yield json.dumps({"type": "error", "data": str(exc)}) + "\n"

    return StreamingResponse(_generator(), media_type="application/x-ndjson")
