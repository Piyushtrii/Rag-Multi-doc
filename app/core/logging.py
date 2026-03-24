import sys
from app.core.config import settings

"""Centralised Loguru configuration.

Import this module once (in ``app/api/main.py``) to configure file sinks.
All other modules keep their ``from loguru import logger`` import as-is.
"""
def setup_logging() -> None:
    """Configure loguru with console + rotating file sink."""
    from loguru import logger

    # Remove the default stderr sink so we control format
    logger.remove()

    #Console sink
    logger.add(
        sys.stderr,
        level=settings.LOG_LEVEL,
        format=(
            "{time:YYYY-MM-DD HH:mm:ss} | "
            "{level: <8} | "
            "{name}:{line} - {message}"
        ),
        colorize=False,
    )

    #Rotating file sink
    settings.LOG_PATH.mkdir(parents=True, exist_ok=True)
    logger.add(
        settings.LOG_PATH / "rag_app.log",
        level=settings.LOG_LEVEL,
        rotation="10 MB",
        retention="14 days",
        compression="zip",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{line} - {message}",
        enqueue=True,  # thread-safe async-friendly
    )
