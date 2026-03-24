import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
"""Tests for the multi-format DocumentLoader."""

#Helpers 

def make_loader(tmp_path: Path):
    """Return a DocumentLoader pointed at *tmp_path*."""
    from app.pipeline.document_loader import DocumentLoader
    loader = DocumentLoader(docs_path=tmp_path)
    return loader


#TXT 

def test_load_txt(tmp_path):
    (tmp_path / "sample.txt").write_text("Hello world. This is a test document.", encoding="utf-8")
    loader = make_loader(tmp_path)
    docs = loader.load_documents()
    assert len(docs) == 1
    assert docs[0]["metadata"]["file_type"] == "txt"
    assert "Hello world" in docs[0]["content"]


#Markdown

def test_load_markdown(tmp_path):
    (tmp_path / "notes.md").write_text("# Title\n\nSome content.", encoding="utf-8")
    loader = make_loader(tmp_path)
    docs = loader.load_documents()
    assert len(docs) == 1
    assert docs[0]["metadata"]["file_type"] == "md"


#Unsupported extension ignored 

def test_unsupported_extension_skipped(tmp_path):
    (tmp_path / "data.csv").write_text("a,b,c", encoding="utf-8")
    loader = make_loader(tmp_path)
    docs = loader.load_documents()
    assert len(docs) == 0


#Non-existent docs path

def test_missing_docs_path(tmp_path):
    from app.pipeline.document_loader import DocumentLoader
    loader = DocumentLoader(docs_path=tmp_path / "does_not_exist")
    docs = loader.load_documents()
    assert docs == []


#Chunking

def test_chunk_documents_produces_chunks(tmp_path):
    # Write a doc long enough to be split
    long_text = "This is sentence number {i}. " * 50
    (tmp_path / "long.txt").write_text(long_text, encoding="utf-8")
    loader = make_loader(tmp_path)
    docs = loader.load_documents()
    chunks = loader.chunk_documents(docs)
    assert len(chunks) > 1
    for chunk in chunks:
        assert "chunk_id" in chunk["metadata"]


def test_chunk_ids_are_unique(tmp_path):
    (tmp_path / "doc1.txt").write_text(("word " * 200), encoding="utf-8")
    (tmp_path / "doc2.txt").write_text(("text " * 200), encoding="utf-8")
    loader = make_loader(tmp_path)
    docs = loader.load_documents()
    chunks = loader.chunk_documents(docs)
    ids = [c["id"] for c in chunks]
    assert len(ids) == len(set(ids)), "Chunk IDs must be unique"
