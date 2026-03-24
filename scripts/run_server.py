import argparse
import uvicorn

from app.core.config import settings


def main() -> None:
    parser = argparse.ArgumentParser(description="Start the Enterprise RAG API server")
    parser.add_argument("--host", default=settings.API_HOST)
    parser.add_argument("--port", type=int, default=settings.API_PORT)
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload (development only)",
    )
    parser.add_argument("--workers", type=int, default=settings.API_WORKERS)
    args = parser.parse_args()

    uvicorn.run(
        "app.api.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        workers=1 if args.reload else args.workers,
        log_level=settings.LOG_LEVEL.lower(),
    )
"""Dev server launcher.

Usage (from the CRAG/ root):
    python -m scripts.run_server
    python -m scripts.run_server --reload
    python -m scripts.run_server --port 9000
"""

if __name__ == "__main__":
    main()
