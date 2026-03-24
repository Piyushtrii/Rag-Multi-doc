import pytest
from pathlib import Path
"""Tests for application configuration."""

def test_settings_import():
    """Settings object should be importable and have required fields."""
    from app.core.config import settings
    assert settings.LLM_PROVIDER in ("groq", "openai", "azure")
    assert settings.CHUNK_SIZE > 0
    assert settings.CHUNK_OVERLAP >= 0
    assert settings.CHUNK_OVERLAP < settings.CHUNK_SIZE


def test_docs_path_is_path_object():
    from app.core.config import settings
    assert isinstance(settings.DOCS_PATH, Path)


def test_db_path_is_path_object():
    from app.core.config import settings
    assert isinstance(settings.DB_PATH, Path)


def test_weights_sum_to_one():
    """BM25 + vector weights should sum to 1.0 (or close enough)."""
    from app.core.config import settings
    total = settings.BM25_WEIGHT + settings.VECTOR_WEIGHT
    assert abs(total - 1.0) < 1e-6, f"Weights sum to {total}, expected 1.0"
