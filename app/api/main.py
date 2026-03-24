from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from app.api.routes import admin, health, query
from app.core.config import settings
from app.core.logging import setup_logging
from app.pipeline.rag_pipeline import RAGPipeline

"""FastAPI application factory.

Run with:
    uvicorn app.api.main:app --reload --port 8000
"""
#Lifespan (replaces deprecated @app.on_event) 
async def lifespan(app: FastAPI):
    """Startup: initialise logging + RAG pipeline. Shutdown: log teardown."""
    setup_logging()
    logger.info("Starting Enterprise RAG Chatbot API")

    try:
        pipeline = RAGPipeline(use_reranking=True, use_hybrid_search=True)
        app.state.rag_pipeline = pipeline
        logger.info("RAG Pipeline attached to app state")
    except Exception as exc:
        logger.error(f"Failed to initialise RAG Pipeline: {exc}")
        raise

    yield  # ← application runs here

    logger.info("Shutting down Enterprise RAG Chatbot API")


#App factory 
def create_app() -> FastAPI:
    app = FastAPI(
        title="Enterprise RAG Chatbot",
        description=(
            "Document Q&A system with multi-format ingestion (TXT, PDF, DOCX, MD), "
            "hybrid BM25+vector search, cross-encoder re-ranking, and streaming responses."
        ),
        version="1.0.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],      # Tighten in production via env var
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    #Routers 
    app.include_router(health.router)
    app.include_router(admin.router)
    app.include_router(query.router)

    return app


app = create_app()


#Dev entry-point
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.api.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=False,
        workers=settings.API_WORKERS,
        log_level=settings.LOG_LEVEL.lower(),
    )
