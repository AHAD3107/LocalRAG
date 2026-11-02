"""Live Ollama integration test for Phase 4.7 Production Retrieval Pipeline.

Verifies end-to-end retrieval and generation with the live running Ollama daemon
and the local persistent VectorIndex on Windows.
"""
from __future__ import annotations

from pathlib import Path
import pytest

from config import settings
from src.embeddings.ollama import OllamaClient
from src.retrieval.pipeline import retrieve_context
from src.storage.vector_index import VectorIndex


@pytest.mark.integration
def test_live_ollama_hybrid_rerank_retrieval():
    client = OllamaClient(settings.ollama_base_url, timeout=30.0)
    if not client.is_available():
        pytest.skip(f"Live Ollama service not reachable on {settings.ollama_base_url}")

    index_dir = settings.index_dir
    index = VectorIndex.load(
        index_dir,
        settings.embedding_model,
        settings.chunk_size,
        settings.chunk_overlap,
    )
    assert len(index.chunks) > 0

    query = "What graph validation library and structure is used in KnowledgeTrace MCQ?"
    matches = retrieve_context(
        query=query,
        index=index,
        client=client,
        top_k=settings.top_k,
        strategy="hybrid_rerank",
        score_threshold=-4.0,
    )

    assert len(matches) > 0
    top_chunk, top_score = matches[0]
    assert "networkx" in top_chunk.text.lower() or "poset" in top_chunk.text.lower()
    assert top_score >= -4.0
    assert top_chunk.source_filename.endswith(".pdf")
    assert top_chunk.page_number >= 1


@pytest.mark.integration
def test_live_ollama_bm25_rerank_retrieval():
    client = OllamaClient(settings.ollama_base_url, timeout=30.0)
    if not client.is_available():
        pytest.skip(f"Live Ollama service not reachable on {settings.ollama_base_url}")

    index_dir = settings.index_dir
    index = VectorIndex.load(
        index_dir,
        settings.embedding_model,
        settings.chunk_size,
        settings.chunk_overlap,
    )

    query = "What is the binary Q-matrix in DINA?"
    matches = retrieve_context(
        query=query,
        index=index,
        client=client,
        top_k=settings.top_k,
        strategy="bm25_rerank",
        score_threshold=-4.0,
    )

    assert len(matches) > 0
    top_chunk, top_score = matches[0]
    assert "q-matrix" in top_chunk.text.lower() or "dina" in top_chunk.text.lower()

