from loguru import logger
from fastapi import APIRouter, HTTPException, Request

"""Health check routes."""

router = APIRouter(tags=["Health"])


@router.get("/")
async def root():
    """API root — lists available endpoints."""
    return {
        "status": "running",
        "service": "Enterprise RAG Chatbot",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "initialize": "/initialize",
            "retrieve": "/retrieve",
            "query": "/query",
            "stream": "/stream",
            "stats": "/stats",
            "docs": "/docs",
        },
    }


@router.get("/health")
async def health_check(request: Request):
    """Liveness / readiness probe."""
    try:
        pipeline = request.app.state.rag_pipeline
        stats = pipeline.get_stats() if pipeline else {}
        return {
            "status": "healthy",
            "rag_pipeline": "initialised" if pipeline else "not initialised",
            "stats": stats,
        }
    except Exception as exc:
        logger.error(f"Health check failed: {exc}")
        raise HTTPException(status_code=500, detail="Service unhealthy")
