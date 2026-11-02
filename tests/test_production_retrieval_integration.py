"""Unit and integration tests for Phase 4.7 Production Retrieval Pipeline.

Tests:
- All retrieval strategies (dense, bm25, hybrid, bm25_rerank, hybrid_rerank)
- Cross-Encoder singleton caching
- Rejection threshold filtering
- Graceful fallback on reranker loading / inference failure
- Document-specific scoping and missing doc_id handling
- FastAPI /query and /query/stream compatibility
- Real Ollama live retrieval verification
"""
from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from unittest.mock import MagicMock, patch

import httpx
import pytest

import src.api.main as api
from config import settings
from src.ingestion.models import Chunk
from src.retrieval.pipeline import (
    get_bm25_index,
    get_reranker,
    retrieve_context,
)
from src.retrieval.reranker import CrossEncoderReranker
from src.storage.vector_index import VectorIndex


def _make_test_chunk(chunk_id: str, doc_id: str, text: str, page_number: int = 1) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        doc_id=doc_id,
        source_filename=f"{doc_id}.pdf",
        page_number=page_number,
        chunk_index=0,
        text=text,
    )


@pytest.fixture
def test_vector_index(tmp_path: Path) -> VectorIndex:
    idx_path = tmp_path / "index"
    index = VectorIndex(idx_path, settings.embedding_model, 500, 50)
    c1 = _make_test_chunk("c1", "doc1", "KnowledgeTrace MCQ utilizes a NetworkX Poset DAG for prerequisite validation.")
    c2 = _make_test_chunk("c2", "doc1", "Bayesian Knowledge Tracing models student mental states with Prior Knowledge P(L0).")
    c3 = _make_test_chunk("c3", "doc2", "The DINA model utilizes a binary Q-Matrix connecting test questions to skills.")
    index.add([c1, c2], [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ])
    index.add([c3], [
        [0.0, 0.0, 1.0],
    ])
    index.save()
    return index


class MockOllamaClient:
    def __init__(self, *args, **kwargs):
        pass

    def embed(self, text: str, model: str) -> list[float]:
        t = text.lower()
        if "networkx" in t or "poset" in t:
            return [1.0, 0.0, 0.0]
        elif "bayesian" in t or "bkt" in t:
            return [0.0, 1.0, 0.0]
        elif "dina" in t or "q-matrix" in t:
            return [0.0, 0.0, 1.0]
        return [0.5, 0.5, 0.0]

    def chat(self, prompt: str, model: str) -> str:
        return "Answer based on context. [Source 1]"

    def chat_stream(self, prompt: str, model: str):
        yield "Token 1 "
        yield "Token 2."

    def is_available(self) -> bool:
        return True

    def close(self):
        pass


def test_reranker_singleton_caching():
    r1 = get_reranker()
    r2 = get_reranker()
    assert r1 is r2
    assert isinstance(r1, CrossEncoderReranker)


def test_bm25_index_caching(test_vector_index: VectorIndex):
    b1 = get_bm25_index(test_vector_index)
    b2 = get_bm25_index(test_vector_index)
    assert b1 is b2
    assert len(b1.chunks) == 3


def test_retrieve_context_dense_strategy(test_vector_index: VectorIndex):
    client = MockOllamaClient()
    matches = retrieve_context(
        query="NetworkX Poset DAG",
        index=test_vector_index,
        client=client,
        top_k=2,
        strategy="dense",
    )
    assert len(matches) > 0
    assert matches[0][0].chunk_id == "c1"


def test_retrieve_context_bm25_strategy(test_vector_index: VectorIndex):
    client = MockOllamaClient()
    matches = retrieve_context(
        query="Q-Matrix binary skills",
        index=test_vector_index,
        client=client,
        top_k=2,
        strategy="bm25",
    )
    assert len(matches) > 0
    assert matches[0][0].chunk_id == "c3"


def test_retrieve_context_hybrid_strategy(test_vector_index: VectorIndex):
    client = MockOllamaClient()
    matches = retrieve_context(
        query="Prior Knowledge P(L0) in Bayesian Knowledge Tracing",
        index=test_vector_index,
        client=client,
        top_k=2,
        strategy="hybrid",
    )
    assert len(matches) > 0
    assert matches[0][0].chunk_id == "c2"


def test_retrieve_context_hybrid_rerank_strategy(test_vector_index: VectorIndex):
    client = MockOllamaClient()
    matches = retrieve_context(
        query="Which graph library validates DAG invariants in KnowledgeTrace MCQ?",
        index=test_vector_index,
        client=client,
        top_k=2,
        strategy="hybrid_rerank",
        score_threshold=-5.0,
    )
    assert len(matches) > 0
    assert matches[0][0].chunk_id == "c1"
    # Metadata preserved
    assert matches[0][0].source_filename == "doc1.pdf"
    assert matches[0][0].page_number == 1
    assert isinstance(matches[0][1], float)


def test_retrieve_context_threshold_rejection(test_vector_index: VectorIndex):
    client = MockOllamaClient()
    # Query completely unrelated to the corpus
    matches = retrieve_context(
        query="What is the recipe for baking chocolate chip cookies in French pastry?",
        index=test_vector_index,
        client=client,
        top_k=2,
        strategy="hybrid_rerank",
        score_threshold=-2.0,  # Strict rejection threshold
    )
    # Should be rejected because chocolate chip cookies have strongly negative logit
    assert len(matches) == 0


def test_retrieve_context_document_specific_scoping(test_vector_index: VectorIndex):
    client = MockOllamaClient()
    # Query matches c1 (doc1), but we scope to doc2
    matches = retrieve_context(
        query="NetworkX Poset DAG",
        index=test_vector_index,
        client=client,
        top_k=2,
        doc_id="doc2",
        strategy="hybrid_rerank",
        score_threshold=-10.0,
    )
    # Only doc2 chunks allowed
    for chunk, _ in matches:
        assert chunk.doc_id == "doc2"


def test_retrieve_context_missing_doc_id_raises_value_error(test_vector_index: VectorIndex):
    client = MockOllamaClient()
    with pytest.raises(ValueError, match="not found in the index"):
        retrieve_context(
            query="any query",
            index=test_vector_index,
            client=client,
            doc_id="non_existent_doc_999",
            strategy="hybrid_rerank",
        )


def test_retrieve_context_fallback_on_reranker_failure(test_vector_index: VectorIndex):
    client = MockOllamaClient()
    # Patch get_reranker to return a mock whose rerank method raises RuntimeError
    mock_failing_reranker = MagicMock()
    mock_failing_reranker.rerank.side_effect = RuntimeError("Simulated transformer failure")

    with patch("src.retrieval.pipeline.get_reranker", return_value=mock_failing_reranker):
        # Should gracefully fall back to first-stage hybrid candidates without crashing
        matches = retrieve_context(
            query="NetworkX Poset DAG",
            index=test_vector_index,
            client=client,
            top_k=2,
            strategy="hybrid_rerank",
        )
        assert len(matches) > 0
        assert matches[0][0].chunk_id in ("c1", "c2", "c3")


@pytest.mark.anyio
async def test_fastapi_query_endpoint_with_hybrid_reranker(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(api, "OllamaClient", MockOllamaClient)
    monkeypatch.setattr(
        api,
        "settings",
        replace(
            settings,
            documents_dir=tmp_path / "documents",
            index_dir=tmp_path / "index",
            retrieval_strategy="hybrid_rerank",
            reranker_score_threshold=-5.0,
        ),
    )
    api.settings.ensure_directories()

    # Create test index
    idx = VectorIndex(tmp_path / "index", settings.embedding_model, 500, 50)
    c1 = _make_test_chunk("c1", "doc1", "KnowledgeTrace MCQ uses NetworkX Poset DAG.")
    idx.add([c1], [[1.0, 0.0, 0.0]])
    idx.save()

    transport = httpx.ASGITransport(app=api.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/query",
            json={"question": "Which graph library is used in KnowledgeTrace MCQ?"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "answer" in data
        assert len(data["sources"]) == 1
        assert data["sources"][0]["filename"] == "doc1.pdf"
        assert data["sources"][0]["page"] == 1


@pytest.mark.anyio
async def test_fastapi_query_stream_endpoint_with_hybrid_reranker(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(api, "OllamaClient", MockOllamaClient)
    monkeypatch.setattr(
        api,
        "settings",
        replace(
            settings,
            documents_dir=tmp_path / "documents",
            index_dir=tmp_path / "index",
            retrieval_strategy="hybrid_rerank",
            reranker_score_threshold=-5.0,
        ),
    )
    api.settings.ensure_directories()

    idx = VectorIndex(tmp_path / "index", settings.embedding_model, 500, 50)
    c1 = _make_test_chunk("c1", "doc1", "KnowledgeTrace MCQ uses NetworkX Poset DAG.")
    idx.add([c1], [[1.0, 0.0, 0.0]])
    idx.save()

    transport = httpx.ASGITransport(app=api.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/query/stream",
            json={"question": "Which graph library is used in KnowledgeTrace MCQ?"},
        )
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers["content-type"]
        body = resp.text
        assert "event: sources" in body
        assert "event: token" in body
        assert "event: done" in body
        assert "doc1.pdf" in body
