import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient
"""FastAPI smoke tests using TestClient with a mocked RAG pipeline."""


#Fixtures

@pytest.fixture()
def mock_pipeline():
    """A minimal mock that satisfies all route calls."""
    p = MagicMock()
    p.get_stats.return_value = {
        "vector_db": {"collection_name": "enterprise_docs", "document_count": 10},
        "use_reranking": True,
        "use_hybrid_search": True,
    }
    p.retrieve.return_value = [
        {
            "id": "doc_chunk_0",
            "content": "Tesla is an energy company.",
            "metadata": {"source": "Tesla.txt"},
            "rerank_score": 0.9,
        }
    ]
    p._prepare_context.return_value = "Tesla is an energy company."
    p.query.return_value = {
        "query": "What is Tesla?",
        "answer": "Tesla is an energy and automotive company.",
        "sources": [
            {
                "id": "doc_chunk_0",
                "source": "Tesla.txt",
                "relevance_score": 0.9,
            }
        ],
    }
    return p


@pytest.fixture()
def client(mock_pipeline):
    """TestClient with pipeline injected into app.state."""
    from app.api.main import app
    
    # Override app lifespan to prevent real pipeline creation during tests
    from contextlib import asynccontextmanager
    @asynccontextmanager
    async def mock_lifespan(app):
        app.state.rag_pipeline = mock_pipeline
        yield
        
    app.router.lifespan_context = mock_lifespan
    
    with TestClient(app, raise_server_exceptions=True) as c:
        yield c


#Health 

def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "healthy"


def test_root(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "endpoints" in r.json()


#Admin 

def test_stats(client):
    r = client.get("/stats")
    assert r.status_code == 200
    assert r.json()["status"] == "success"


def test_initialize(client, mock_pipeline):
    r = client.post("/initialize", json={"force_reload": False})
    assert r.status_code == 200
    mock_pipeline.initialize_database.assert_called_once()


#Query
def test_query_empty_string_errors(client):
    r = client.post("/query", json={"query": "   "})
    assert r.status_code == 400


def test_query_success(client):
    r = client.post("/query", json={"query": "What is Tesla?", "k": 3})
    assert r.status_code == 200
    data = r.json()
    assert "answer" in data
    assert data["query"] == "What is Tesla?"


def test_retrieve_success(client):
    r = client.post("/retrieve", json={"query": "Tesla mission", "k": 3})
    assert r.status_code == 200
    data = r.json()
    assert data["total_results"] >= 1
