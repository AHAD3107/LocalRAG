"""Tests for multi-document isolation, scoping boundaries, and lifecycle safety.

Verifies Task 3 requirements:
1. Scoped retrieval to Doc A returns zero chunks from Doc B, C, D, or E.
2. Cross-document queries return chunks only from relevant documents.
3. Querying a non-existent document ID raises a clear error rather than returning incorrect chunks.
4. Document deletion removes vectors from VectorIndex, removes entries from BM25 index,
   causes scoped queries to fail gracefully, and prevents deleted chunks from appearing in cross-doc retrieval.
5. Re-indexing a document does not create duplicate entries (raises ValueError / 409).
6. Vector index persistence across reload preserves all document boundaries, IDs, and page metadata.
"""
from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from unittest.mock import MagicMock

import httpx
import numpy as np
import pytest

import src.api.main as api
from config import settings
from src.ingestion.models import Chunk
from src.retrieval.bm25_search import BM25Index
from src.retrieval.hybrid_search import hybrid_search
from src.retrieval.pipeline import retrieve_context
from src.retrieval.search import search as dense_search
from src.storage.vector_index import VectorIndex


class MockOllamaClient:
    def __init__(self, *args, **kwargs):
        pass

    def check_health(self) -> bool:
        return True

    def embed(self, text: str, model: str | None = None) -> list[float]:
        return [1.0, 0.0, 0.0, 0.0]

    def generate(self, *args, **kwargs) -> str:
        return "Mock response"


def _create_mock_chunk(doc_id: str, page_num: int, chunk_idx: int, text: str) -> Chunk:
    return Chunk(
        chunk_id=f"{doc_id}_p{page_num}_c{chunk_idx}",
        doc_id=doc_id,
        source_filename=f"doc_{doc_id}.pdf",
        page_number=page_num,
        chunk_index=chunk_idx,
        text=text,
    )


def test_scoped_retrieval_strict_document_boundary():
    """Verify that scoped search to Doc A NEVER returns chunks from Doc B, C, D, or E."""
    doc_a = "doc_alpha_111"
    doc_b = "doc_beta_222"
    doc_c = "doc_gamma_333"

    chunks_a = [
        _create_mock_chunk(doc_a, 1, 0, "B+ Tree index with fanout 512 for database tables."),
        _create_mock_chunk(doc_a, 1, 1, "Write-Ahead Logging ARIES crash recovery."),
    ]
    chunks_b = [
        _create_mock_chunk(doc_b, 1, 0, "TCP 3-way handshake SYN SYN-ACK ACK connection setup."),
        _create_mock_chunk(doc_b, 2, 1, "TCP sliding window flow control and AIMD congestion."),
    ]
    chunks_c = [
        _create_mock_chunk(doc_c, 1, 0, "CPython memory management with PyMalloc 4KB pools."),
    ]

    dim = 8
    vecs_a = [[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.9, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]
    vecs_b = [[0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.9, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0]]
    vecs_c = [[0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]]

    # Build mock index in memory
    index = MagicMock()
    index.chunks = chunks_a + chunks_b + chunks_c
    index.vectors = np.asarray(vecs_a + vecs_b + vecs_c, dtype=np.float32)

    # 1. Test Dense Retrieval Scoping
    query_vec = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    results_a = dense_search(index, query_vec, top_k=5, threshold=0.1, doc_id=doc_a)
    assert len(results_a) > 0
    assert all(c.doc_id == doc_a for c, _ in results_a)
    assert not any(c.doc_id in (doc_b, doc_c) for c, _ in results_a)

    # 2. Test BM25 Retrieval Scoping
    bm25 = BM25Index(index.chunks)
    bm25_res_b = bm25.search("TCP handshake connection", top_k=5, min_score=0.1, doc_id=doc_b)
    assert len(bm25_res_b) > 0
    assert all(c.doc_id == doc_b for c, _ in bm25_res_b)
    assert not any(c.doc_id in (doc_a, doc_c) for c, _ in bm25_res_b)

    # 3. Test Hybrid Retrieval Scoping
    hybrid_res, _ = hybrid_search(
        index=index,
        query="TCP handshake",
        query_vector=[0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        bm25_index=bm25,
        top_k=5,
        dense_threshold=0.1,
        bm25_min_score=0.1,
        doc_id=doc_b,
    )
    assert len(hybrid_res) > 0
    assert all(c.doc_id == doc_b for c, _ in hybrid_res)


def test_querying_nonexistent_doc_id_raises_clear_error():
    """Verify that querying a non-existent document ID raises ValueError rather than returning wrong chunks."""
    doc_a = "doc_alpha_111"
    chunks_a = [_create_mock_chunk(doc_a, 1, 0, "Database storage engine specification.")]
    index = MagicMock()
    index.chunks = chunks_a
    index.vectors = np.asarray([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)

    # Dense Search with invalid doc_id
    with pytest.raises(ValueError, match="was not found in the index"):
        dense_search(index, [1.0, 0.0, 0.0, 0.0], doc_id="nonexistent_doc_999")

    # BM25 Search with invalid doc_id
    bm25 = BM25Index(index.chunks)
    with pytest.raises(ValueError, match="was not found in the index"):
        bm25.search("storage engine", doc_id="nonexistent_doc_999")


@pytest.mark.anyio
async def test_api_invalid_doc_id_returns_404(tmp_path: Path, monkeypatch):
    """Verify FastAPI /query endpoint maps non-existent doc_id to HTTP 404."""
    monkeypatch.setattr(api, "OllamaClient", MockOllamaClient)
    monkeypatch.setattr(
        api,
        "settings",
        replace(
            settings,
            documents_dir=tmp_path / "documents",
            index_dir=tmp_path / "index",
        ),
    )
    api.settings.ensure_directories()

    idx = VectorIndex(tmp_path / "index", settings.embedding_model, 500, 50)
    c1 = _create_mock_chunk("doc_real_1", 1, 0, "Real document content.")
    idx.add([c1], [[1.0, 0.0, 0.0, 0.0]])
    idx.save()

    transport = httpx.ASGITransport(app=api.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/query",
            json={"question": "What is in the document?", "doc_id": "missing_doc_xyz_404"},
        )
        assert resp.status_code == 404
        detail = resp.json().get("detail", "")
        assert "was not found in the index" in detail or "missing_doc_xyz_404" in detail


def test_document_deletion_lifecycle_and_bm25_sync(tmp_path: Path):
    """Verify deleting a document clears its vectors and chunks from both VectorIndex and BM25 index."""
    idx_dir = tmp_path / "index"
    idx_dir.mkdir(parents=True)

    index = VectorIndex(idx_dir, embedding_model="nomic-embed-text", chunk_size=500, chunk_overlap=50)

    doc1_id = "doc_stay_001"
    doc2_id = "doc_delete_002"

    chunks_doc1 = [
        _create_mock_chunk(doc1_id, 1, 0, "Python GIL Global Interpreter Lock mechanism."),
        _create_mock_chunk(doc1_id, 1, 1, "PyMalloc memory allocator manages 4KB pools."),
    ]
    chunks_doc2 = [
        _create_mock_chunk(doc2_id, 1, 0, "Raft consensus leader election randomized timeout."),
        _create_mock_chunk(doc2_id, 2, 1, "Two-Phase Commit distributed transaction coordinator."),
    ]

    vecs_doc1 = [[1.0, 0.0, 0.0, 0.0], [0.9, 0.1, 0.0, 0.0]]
    vecs_doc2 = [[0.0, 1.0, 0.0, 0.0], [0.0, 0.9, 0.1, 0.0]]

    # Index both documents
    index.add(chunks_doc1, vecs_doc1)
    index.add(chunks_doc2, vecs_doc2)
    index.save()

    assert len(index.chunks) == 4
    assert index.vectors.shape[0] == 4

    # Build BM25 index and verify doc2 is present
    bm25 = BM25Index(index.chunks)
    res_before = bm25.search("Raft consensus leader", top_k=2)
    assert any(c.doc_id == doc2_id for c, _ in res_before)

    # Delete doc2
    removed = index.remove_document(doc2_id)
    assert removed == 2
    index.save()

    # Verify VectorIndex updated
    assert len(index.chunks) == 2
    assert index.vectors.shape[0] == 2
    assert all(c.doc_id == doc1_id for c in index.chunks)

    # Scoped query to deleted doc2 raises ValueError
    with pytest.raises(ValueError, match="was not found in the index"):
        dense_search(index, [0.0, 1.0, 0.0, 0.0], doc_id=doc2_id)

    # Rebuild BM25 from updated index chunks and verify doc2 is gone
    bm25_after = BM25Index(index.chunks)
    res_after = bm25_after.search("Raft consensus leader", top_k=2)
    assert not any(c.doc_id == doc2_id for c, _ in res_after)
    with pytest.raises(ValueError, match="was not found in the index"):
        bm25_after.search("Raft consensus leader", doc_id=doc2_id)


def test_reindexing_duplicate_document_rejected(tmp_path: Path):
    """Verify attempting to add an already indexed document raises ValueError."""
    idx_dir = tmp_path / "index"
    idx_dir.mkdir(parents=True)

    index = VectorIndex(idx_dir, embedding_model="nomic-embed-text", chunk_size=500, chunk_overlap=50)

    doc_id = "doc_duplicate_test"
    chunks = [_create_mock_chunk(doc_id, 1, 0, "Test chunk content for duplicate prevention.")]
    vecs = [[0.5, 0.5, 0.5, 0.5]]

    index.add(chunks, vecs)
    index.save()

    # Re-adding the same doc_id must raise ValueError("document is already indexed")
    with pytest.raises(ValueError, match="document is already indexed"):
        index.add(chunks, vecs)


def test_vector_index_persistence_multidoc_integrity(tmp_path: Path):
    """Verify that multiple documents persist across index reload with exact metadata preservation."""
    idx_dir = tmp_path / "index"
    idx_dir.mkdir(parents=True)

    index = VectorIndex(idx_dir, embedding_model="nomic-embed-text", chunk_size=500, chunk_overlap=50)

    doc_ids = ["doc_ml", "doc_db", "doc_net", "doc_py", "doc_se"]
    total_chunks = []

    for i, did in enumerate(doc_ids):
        doc_chunks = [
            _create_mock_chunk(did, 1, 0, f"Page 1 content for document {did}"),
            _create_mock_chunk(did, 2, 1, f"Page 2 content for document {did}"),
        ]
        vecs = [[float(j == i) for j in range(len(doc_ids))] for _ in range(2)]
        index.add(doc_chunks, vecs)
        total_chunks.extend(doc_chunks)

    index.save()
    assert len(index.chunks) == 10

    # Reload from disk into a fresh VectorIndex instance
    reloaded = VectorIndex.load(idx_dir, embedding_model="nomic-embed-text", chunk_size=500, chunk_overlap=50)

    assert len(reloaded.chunks) == 10
    assert reloaded.vectors.shape == (10, 5)

    # Verify every chunk's id, doc_id, source_filename, and page_number matches
    for orig, loaded in zip(total_chunks, reloaded.chunks):
        assert orig.chunk_id == loaded.chunk_id
        assert orig.doc_id == loaded.doc_id
        assert orig.source_filename == loaded.source_filename
        assert orig.page_number == loaded.page_number
        assert orig.text == loaded.text
