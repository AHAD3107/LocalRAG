"""Automated tests for experimental BM25 sparse retrieval."""
from __future__ import annotations

import pytest

from src.ingestion.models import Chunk
from src.retrieval.bm25_search import BM25Index, bm25_search, tokenize


def test_tokenize_preserves_technical_compounds():
    text = "Inference speed < 15ms with P(L0)=0.20, Q-Matrix, and networkx.DiGraph."
    tokens = tokenize(text)

    # Check that compound tokens are preserved
    assert "p(l0)" in tokens
    assert "q-matrix" in tokens
    assert "networkx.digraph" in tokens
    assert "0.20" in tokens
    assert "15ms" in tokens

    # Check that sub-tokens are also emitted for flexible matching
    assert "l0" in tokens
    assert "matrix" in tokens
    assert "networkx" in tokens
    assert "digraph" in tokens


def test_tokenize_empty_and_whitespace():
    assert tokenize("") == []
    assert tokenize("   \n\t  ") == []


def test_bm25_empty_index():
    index = BM25Index([])
    assert index.search("anything") == []
    with pytest.raises(ValueError, match="Document 'doc1' was not found"):
        index.search("anything", doc_id="doc1")


def test_bm25_empty_query():
    chunk = Chunk("c1", "doc1", "f.pdf", 1, 0, "Some sample technical text.")
    index = BM25Index([chunk])
    assert index.search("") == []
    assert index.search("    ") == []


def test_bm25_exact_technical_token_match():
    c1 = Chunk("c1", "doc1", "f.pdf", 1, 0, "Prior knowledge probability P(L0) is set to 0.20.")
    c2 = Chunk("c2", "doc1", "f.pdf", 2, 1, "Learning transition probability P(T) is set to 0.15.")
    c3 = Chunk("c3", "doc1", "f.pdf", 3, 2, "Careless slip probability P(S) is set to 0.10.")
    index = BM25Index([c1, c2, c3])

    results = index.search("What is P(L0)?", top_k=3)
    assert len(results) > 0
    assert results[0][0].chunk_id == "c1"
    assert results[0][1] > 0.0

    results_pt = index.search("What is P(T)?", top_k=3)
    assert results_pt[0][0].chunk_id == "c2"


def test_bm25_doc_id_filter():
    c1 = Chunk("c1", "doc_a", "a.pdf", 1, 0, "FastAPI backend architecture.")
    c2 = Chunk("c2", "doc_b", "b.pdf", 1, 0, "FastAPI backend database architecture.")
    index = BM25Index([c1, c2])

    # Unfiltered
    unfiltered = index.search("FastAPI", top_k=2)
    assert len(unfiltered) == 2

    # Filtered to doc_a
    filtered = index.search("FastAPI", top_k=2, doc_id="doc_a")
    assert len(filtered) == 1
    assert filtered[0][0].chunk_id == "c1"
    assert filtered[0][0].doc_id == "doc_a"

    # Nonexistent doc_id raises ValueError
    with pytest.raises(ValueError, match="Document 'missing_doc' was not found"):
        index.search("FastAPI", doc_id="missing_doc")


def test_bm25_min_score_cutoff():
    c1 = Chunk("c1", "doc1", "f.pdf", 1, 0, "NetworkX directed acyclic graph Kahn algorithm.")
    index = BM25Index([c1])

    # Normal search retrieves with score
    res = index.search("NetworkX", min_score=0.0)
    assert len(res) == 1
    score = res[0][1]

    # Cutoff above score rejects
    res_cutoff = index.search("NetworkX", min_score=score + 10.0)
    assert res_cutoff == []


def test_bm25_duplicate_chunks():
    c1 = Chunk("c1", "doc1", "f.pdf", 1, 0, "NetworkX directed graph.")
    c2 = Chunk("c2", "doc1", "f.pdf", 1, 1, "NetworkX directed graph.")
    index = BM25Index([c1, c2])

    results = index.search("NetworkX", top_k=5)
    assert len(results) == 2
    assert results[0][0].chunk_id == "c1"
    assert results[1][0].chunk_id == "c2"
    assert pytest.approx(results[0][1]) == results[1][1]


def test_bm25_search_convenience_function():
    chunks = [
        Chunk("c1", "doc1", "f.pdf", 1, 0, "Cognitive diagnosis Q-Matrix logic."),
        Chunk("c2", "doc1", "f.pdf", 2, 1, "Bayesian knowledge tracing HMM."),
    ]
    results = bm25_search(chunks, "Q-Matrix", top_k=1)
    assert len(results) == 1
    assert results[0][0].chunk_id == "c1"

