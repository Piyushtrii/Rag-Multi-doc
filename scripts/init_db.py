import argparse
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description="Initialise the RAG vector database")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Delete all existing vectors before loading documents",
    )
    args = parser.parse_args()

    # Import here so settings + logging are available after sys.path is set
    from app.core.logging import setup_logging
    from app.pipeline.rag_pipeline import RAGPipeline

    setup_logging()

    from loguru import logger

    logger.info("Starting database initialisation script")
    pipeline = RAGPipeline(use_reranking=True, use_hybrid_search=True)

    if args.force:
        logger.warning("Force flag set — clearing existing vector DB")
        pipeline.vector_db.delete_all()

    pipeline.initialize_database()
    stats = pipeline.get_stats()
    logger.info(f"Done! Stats: {stats}")
    print(f"\n✅ Database ready — {stats['vector_db']['document_count']} chunks indexed")


if __name__ == "__main__":
    main()
