"""Admin routes — database initialisation and system statistics."""

from fastapi import APIRouter, HTTPException, Request
from loguru import logger

from app.api.schemas import InitializeRequest

router = APIRouter(tags=["Admin"])


@router.post("/initialize")
async def initialize(request: Request, body: InitializeRequest = InitializeRequest()):
    """Initialise or reload the vector database from the docs folder."""
    pipeline = request.app.state.rag_pipeline
    try:
        logger.info(f"Initialising database (force_reload={body.force_reload})")
        if body.force_reload:
            pipeline.vector_db.delete_all()
        pipeline.initialize_database()
        return {
            "status": "success",
            "message": "Database initialised successfully",
            "stats": pipeline.get_stats(),
        }
    except Exception as exc:
        logger.error(f"Database initialisation failed: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/stats")
async def get_stats(request: Request):
    """Return system and vector-DB statistics."""
    pipeline = request.app.state.rag_pipeline
    try:
        return {"status": "success", "stats": pipeline.get_stats()}
    except Exception as exc:
        logger.error(f"Stats retrieval failed: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))
