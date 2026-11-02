"""Unit tests for CrossEncoderReranker in src.retrieval.reranker."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.ingestion.models import Chunk
from src.retrieval.reranker import CrossEncoderReranker, RerankCandidateDetail, _sigmoid


def _make_chunk(
    chunk_id: str,
    text: str,
    page_number: int = 1,
    doc_id: str = "doc_1",
    chunk_index: int = 0,
    source_filename: str = "test_doc.pdf",
) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        doc_id=doc_id,
        source_filename=source_filename,
        page_number=page_number,
        chunk_index=chunk_index,
        text=text,
    )


def test_sigmoid_values():
    assert _sigmoid(0.0) == pytest.approx(0.5)
    assert _sigmoid(100.0) == pytest.approx(1.0)
    assert _sigmoid(-100.0) == pytest.approx(0.0)


def test_lazy_loading_does_not_initialize_model_immediately():
    reranker = CrossEncoderReranker(lazy_load=True)
    assert reranker.is_loaded is False


def test_rerank_empty_candidates():
    reranker = CrossEncoderReranker(lazy_load=True)
    res = reranker.rerank("any query", [])
    assert res == []


def test_rerank_zero_or_negative_top_k():
    chunk = _make_chunk("c1", "some text")
    reranker = CrossEncoderReranker(lazy_load=True)
    res = reranker.rerank("any query", [(chunk, 0.9)], top_k=0)
    assert res == []

    res_neg = reranker.rerank("any query", [(chunk, 0.9)], top_k=-1)
    assert res_neg == []


def test_rerank_sorts_by_cross_encoder_score():
    chunk_a = _make_chunk("c_a", "low relevance text")
    chunk_b = _make_chunk("c_b", "high relevance text")
    candidates = [(chunk_a, 0.9), (chunk_b, 0.5)]  # original order: a then b

    reranker = CrossEncoderReranker(lazy_load=True)
    # Mock score_pairs so that chunk_b gets higher score than chunk_a
    with patch.object(reranker, "score_pairs", return_value=[1.2, 5.8]):
        reranked = reranker.rerank("test query", candidates, top_k=2)

    assert len(reranked) == 2
    assert reranked[0][0].chunk_id == "c_b"
    assert reranked[0][1] == pytest.approx(5.8)
    assert reranked[1][0].chunk_id == "c_a"
    assert reranked[1][1] == pytest.approx(1.2)


def test_rerank_preserves_all_metadata():
    chunk = _make_chunk(
        chunk_id="chk_42",
        text="specific technical detail",
        page_number=7,
        doc_id="doc_xyz",
        chunk_index=3,
        source_filename="manual.pdf",
    )
    reranker = CrossEncoderReranker(lazy_load=True)
    with patch.object(reranker, "score_pairs", return_value=[4.5]):
        results = reranker.rerank("detail query", [(chunk, 0.7)], top_k=1)

    assert len(results) == 1
    out_chunk, out_score = results[0]
    assert out_chunk.chunk_id == "chk_42"
    assert out_chunk.doc_id == "doc_xyz"
    assert out_chunk.page_number == 7
    assert out_chunk.chunk_index == 3
    assert out_chunk.source_filename == "manual.pdf"
    assert out_chunk.text == "specific technical detail"
    assert out_score == pytest.approx(4.5)


def test_rerank_score_threshold_filtering():
    chunk_a = _make_chunk("c_a", "text a")
    chunk_b = _make_chunk("c_b", "text b")
    chunk_c = _make_chunk("c_c", "text c")
    candidates = [(chunk_a, 0.8), (chunk_b, 0.7), (chunk_c, 0.6)]

    reranker = CrossEncoderReranker(lazy_load=True)
    # Logit scores: 2.0, -1.5, -5.0
    with patch.object(reranker, "score_pairs", return_value=[2.0, -1.5, -5.0]):
        # Filter with threshold = 0.0
        results = reranker.rerank("query", candidates, top_k=3, score_threshold=0.0)

    assert len(results) == 1
    assert results[0][0].chunk_id == "c_a"
    assert results[0][1] == pytest.approx(2.0)


def test_rerank_fallback_on_scoring_failure():
    chunk_a = _make_chunk("c_a", "text a")
    chunk_b = _make_chunk("c_b", "text b")
    candidates = [(chunk_a, 0.95), (chunk_b, 0.85)]

    reranker = CrossEncoderReranker(lazy_load=True)
    with patch.object(reranker, "score_pairs", side_effect=RuntimeError("CUDA out of memory")):
        results = reranker.rerank("query", candidates, top_k=2)

    # Should fall back cleanly to original first-stage candidates
    assert len(results) == 2
    assert results[0][0].chunk_id == "c_a"
    assert results[0][1] == pytest.approx(0.95)
    assert results[1][0].chunk_id == "c_b"
    assert results[1][1] == pytest.approx(0.85)


def test_rerank_report_details():
    chunk_a = _make_chunk("c_a", "text a")
    chunk_b = _make_chunk("c_b", "text b")
    candidates = [(chunk_a, 0.6), (chunk_b, 0.8)]

    reranker = CrossEncoderReranker(lazy_load=True, device="cpu")
    with patch.object(reranker, "score_pairs", return_value=[3.0, 1.0]):
        report = reranker.rerank_with_report("query", candidates, top_k=2)

    assert report.total_candidates == 2
    assert report.returned_candidates == 2
    assert report.device == "cpu"
    assert len(report.details) == 2
    # chunk_a scored 3.0, was rank 1 in input
    assert report.details[0].chunk_id == "c_a"
    assert report.details[0].original_rank == 1
    assert report.details[0].original_score == 0.6
    assert report.details[0].rerank_score == pytest.approx(3.0)
    assert report.details[0].sigmoid_score == pytest.approx(_sigmoid(3.0))


def test_device_selection_cpu():
    reranker = CrossEncoderReranker(device="cpu", lazy_load=True)
    assert reranker.device == "cpu"
