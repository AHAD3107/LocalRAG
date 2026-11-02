"""Automated tests for experimental Reciprocal Rank Fusion (RRF) hybrid search."""
from __future__ import annotations

from pathlib import Path
import pytest

from src.ingestion.models import Chunk
from src.retrieval.bm25_search import BM25Index
from src.retrieval.hybrid_search import hybrid_search, reciprocal_rank_fusion
from src.storage.vector_index import VectorIndex


def test_reciprocal_rank_fusion_math():
    c1 = Chunk("c1", "doc1", "f.pdf", 1, 0, "Chunk one text.")
    c2 = Chunk("c2", "doc1", "f.pdf", 1, 1, "Chunk two text.")
    c3 = Chunk("c3", "doc1", "f.pdf", 1, 2, "Chunk three text.")

    # Dense ranks: c1 (rank 1), c2 (rank 2)
    dense_results = [(c1, 0.90), (c2, 0.80)]
    # BM25 ranks: c2 (rank 1), c3 (rank 2)
    bm25_results = [(c2, 12.0), (c3, 8.0)]

    # RRF with k=60:
    # c1: 1/(60+1) + 0 = 1/61 ~ 0.0163934
    # c2: 1/(60+2) + 1/(60+1) = 1/62 + 1/61 ~ 0.0161290 + 0.0163934 = 0.0325225
    # c3: 0 + 1/(60+2) = 1/62 ~ 0.0161290
    results, details = reciprocal_rank_fusion(
        dense_results=dense_results,
        bm25_results=bm25_results,
        rrf_k=60,
        dense_weight=1.0,
        bm25_weight=1.0,
        top_k=3,
    )

    # c2 must be top rank because of corroboration across both dense and sparse rankings!
    assert len(results) == 3
    assert results[0][0].chunk_id == "c2"
    assert results[1][0].chunk_id == "c1"
    assert results[2][0].chunk_id == "c3"

    assert pytest.approx(results[0][1], rel=1e-4) == (1 / 62 + 1 / 61)
    assert pytest.approx(results[1][1], rel=1e-4) == (1 / 61)
    assert pytest.approx(results[2][1], rel=1e-4) == (1 / 62)

    # Verify detail reporting
    c2_detail = next(d for d in details if d.chunk_id == "c2")
    assert c2_detail.dense_rank == 2
    assert c2_detail.dense_score == 0.80
    assert c2_detail.bm25_rank == 1
    assert c2_detail.bm25_score == 12.0


def test_reciprocal_rank_fusion_one_sided():
    c1 = Chunk("c1", "doc1", "f.pdf", 1, 0, "Chunk one.")
    # BM25 empty
    results_dense_only, _ = reciprocal_rank_fusion([(c1, 0.85)], [], rrf_k=60)
    assert len(results_dense_only) == 1
    assert results_dense_only[0][0].chunk_id == "c1"
    assert pytest.approx(results_dense_only[0][1], rel=1e-4) == (1 / 61)

    # Dense empty
    results_bm25_only, _ = reciprocal_rank_fusion([], [(c1, 10.0)], rrf_k=60)
    assert len(results_bm25_only) == 1
    assert results_bm25_only[0][0].chunk_id == "c1"
    assert pytest.approx(results_bm25_only[0][1], rel=1e-4) == (1 / 61)

    # Both empty
    results_none, details_none = reciprocal_rank_fusion([], [])
    assert results_none == []
    assert details_none == []


def test_hybrid_search_end_to_end(tmp_path: Path):
    c1 = Chunk("c1", "doc1", "f.pdf", 1, 0, "NetworkX directed acyclic graph Kahn algorithm.")
    c2 = Chunk("c2", "doc1", "f.pdf", 2, 1, "Bayesian knowledge tracing HMM hidden markov model.")

    v_idx = VectorIndex(tmp_path, "embed", 50, 0)
    v_idx.add([c1, c2], [[1.0, 0.0], [0.0, 1.0]])

    bm25_idx = BM25Index([c1, c2])

    # Query matching c1 both lexically and semantically
    fused, report = hybrid_search(
        index=v_idx,
        query="NetworkX graph algorithm",
        query_vector=[1.0, 0.0],
        bm25_index=bm25_idx,
        top_k=2,
        dense_threshold=0.5,
    )

    assert len(fused) > 0
    assert fused[0][0].chunk_id == "c1"
    assert report.dense_count > 0
    assert report.bm25_count > 0


def test_hybrid_search_doc_id_filtering(tmp_path: Path):
    c1 = Chunk("c1", "doc_a", "a.pdf", 1, 0, "FastAPI asynchronous background tasks.")
    c2 = Chunk("c2", "doc_b", "b.pdf", 1, 0, "FastAPI asynchronous background tasks.")

    v_idx = VectorIndex(tmp_path, "embed", 50, 0)
    v_idx.add([c1], [[1.0, 0.0]])
    v_idx.add([c2], [[0.9, 0.1]])

    fused, _ = hybrid_search(
        index=v_idx,
        query="FastAPI",
        query_vector=[1.0, 0.0],
        doc_id="doc_a",
        top_k=2,
        dense_threshold=0.5,
    )
    assert len(fused) == 1
    assert fused[0][0].doc_id == "doc_a"

    # Missing doc_id raises ValueError
    with pytest.raises(ValueError, match="Document 'nonexistent' was not found"):
        hybrid_search(
            index=v_idx,
            query="FastAPI",
            query_vector=[1.0, 0.0],
            doc_id="nonexistent",
        )

