"""Configurable production retrieval pipeline for LocalRAG.

Supports multiple selectable retrieval strategies:
- "hybrid_rerank" (Default): Hybrid RRF candidate generation + Cross-Encoder reranking
- "bm25_rerank": Fast sparse BM25 candidate generation + Cross-Encoder reranking
- "hybrid": Dense (cosine similarity) + Sparse (BM25) with Reciprocal Rank Fusion
- "bm25": Sparse-only Okapi BM25 retrieval
- "dense": Semantic vector cosine similarity retrieval (original baseline / rollback option)

Provides singleton caching for the Cross-Encoder and BM25 index, preserves all Chunk metadata,
and guarantees graceful fallback if the reranker fails or cannot load.
"""
from __future__ import annotations

import logging
import threading
from typing import Sequence

from config import settings
from src.embeddings.ollama import OllamaClient
from src.ingestion.models import Chunk
from src.retrieval.bm25_search import BM25Index
from src.retrieval.hybrid_search import hybrid_search
from src.retrieval.reranker import CrossEncoderReranker
from src.retrieval.search import search as dense_search
from src.storage.vector_index import VectorIndex

logger = logging.getLogger(__name__)

# Module-level singletons / caches with thread safety
_RERANKER_LOCK = threading.Lock()
_RERANKER_INSTANCE: CrossEncoderReranker | None = None
_BM25_LOCK = threading.Lock()
_CACHED_BM25_INDEX: BM25Index | None = None
_CACHED_BM25_CHUNKS_HASH: int | None = None


def get_reranker(
    model_name: str | None = None,
    device: str | None = None,
) -> CrossEncoderReranker:
    """Return the cached CrossEncoderReranker singleton, initializing lazily if needed."""
    global _RERANKER_INSTANCE
    target_model = model_name or settings.reranker_model
    target_device = device or settings.reranker_device

    if _RERANKER_INSTANCE is None or _RERANKER_INSTANCE.model_name != target_model:
        with _RERANKER_LOCK:
            if _RERANKER_INSTANCE is None or _RERANKER_INSTANCE.model_name != target_model:
                logger.info(
                    "Initializing CrossEncoderReranker singleton (model: %s, device: %s)",
                    target_model,
                    target_device,
                )
                _RERANKER_INSTANCE = CrossEncoderReranker(
                    model_name=target_model,
                    device=target_device,
                    lazy_load=False,
                )
    return _RERANKER_INSTANCE


def get_bm25_index(index: VectorIndex) -> BM25Index:
    """Return a cached BM25Index for the loaded VectorIndex, rebuilding only if chunks changed."""
    global _CACHED_BM25_INDEX, _CACHED_BM25_CHUNKS_HASH
    # Hash check on chunk_ids to detect updates
    current_hash = hash(tuple(c.chunk_id for c in index.chunks))
    if _CACHED_BM25_INDEX is None or _CACHED_BM25_CHUNKS_HASH != current_hash:
        with _BM25_LOCK:
            if _CACHED_BM25_INDEX is None or _CACHED_BM25_CHUNKS_HASH != current_hash:
                logger.info("Building in-memory BM25Index for %d indexed chunks", len(index.chunks))
                _CACHED_BM25_INDEX = BM25Index(index.chunks, k1=1.5, b=0.75)
                _CACHED_BM25_CHUNKS_HASH = current_hash
    return _CACHED_BM25_INDEX


def retrieve_context(
    query: str,
    index: VectorIndex,
    client: OllamaClient,
    top_k: int | None = None,
    doc_id: str | None = None,
    strategy: str | None = None,
    score_threshold: float | None = None,
) -> list[tuple[Chunk, float]]:
    """
    Execute configured retrieval strategy and return ranked (Chunk, score) pairs.

    Args:
        query: User query string (contextualized if conversational).
        index: Loaded VectorIndex containing persistent document chunks and embeddings.
        client: OllamaClient for query embedding generation.
        top_k: Number of final chunks to return (defaults to settings.top_k).
        doc_id: Optional document identifier to scope search to a single document.
        strategy: Retrieval strategy override ("hybrid_rerank", "hybrid", "bm25_rerank", "bm25", "dense").
        score_threshold: Optional cutoff for reranker or similarity score.

    Returns:
        List of (Chunk, score) tuples sorted descending by score.
    """
    active_strategy = (strategy or settings.retrieval_strategy).lower().strip()
    k = top_k if top_k is not None else settings.top_k
    first_stage_depth = max(k, settings.reranker_first_stage_depth)
    threshold = score_threshold if score_threshold is not None else settings.reranker_score_threshold

    if active_strategy == "dense":
        query_vector = client.embed(query, settings.embedding_model)
        dense_thresh = score_threshold if score_threshold is not None else settings.similarity_threshold
        return dense_search(
            index,
            query_vector,
            top_k=k,
            threshold=dense_thresh,
            doc_id=doc_id,
        )

    elif active_strategy == "bm25":
        bm25_idx = get_bm25_index(index)
        return bm25_idx.search(
            query=query,
            top_k=k,
            min_score=3.0,
            doc_id=doc_id,
        )

    elif active_strategy == "hybrid":
        query_vector = client.embed(query, settings.embedding_model)
        bm25_idx = get_bm25_index(index)
        results, _ = hybrid_search(
            index=index,
            query=query,
            query_vector=query_vector,
            bm25_index=bm25_idx,
            top_k=k,
            dense_threshold=settings.similarity_threshold,
            bm25_min_score=3.0,
            doc_id=doc_id,
            rrf_k=60,
        )
        return results

    elif active_strategy == "bm25_rerank":
        bm25_idx = get_bm25_index(index)
        bm25_cands = bm25_idx.search(
            query=query,
            top_k=first_stage_depth,
            min_score=3.0,
            doc_id=doc_id,
        )
        if not bm25_cands:
            return []

        try:
            reranker = get_reranker()
            return reranker.rerank(
                query=query,
                candidates=bm25_cands,
                top_k=k,
                score_threshold=threshold,
            )
        except Exception as exc:
            logger.warning(
                "CrossEncoder reranking failed (%s); falling back to BM25 first-stage candidates.",
                exc,
            )
            return bm25_cands[:k]

    elif active_strategy == "hybrid_rerank":
        query_vector = client.embed(query, settings.embedding_model)
        bm25_idx = get_bm25_index(index)
        hybrid_cands, _ = hybrid_search(
            index=index,
            query=query,
            query_vector=query_vector,
            bm25_index=bm25_idx,
            top_k=first_stage_depth,
            dense_threshold=settings.similarity_threshold,
            bm25_min_score=3.0,
            doc_id=doc_id,
            rrf_k=60,
        )
        if not hybrid_cands:
            return []

        try:
            reranker = get_reranker()
            return reranker.rerank(
                query=query,
                candidates=hybrid_cands,
                top_k=k,
                score_threshold=threshold,
            )
        except Exception as exc:
            logger.warning(
                "CrossEncoder reranking failed (%s); falling back to hybrid first-stage candidates.",
                exc,
            )
            return hybrid_cands[:k]

    else:
        logger.warning("Unrecognized retrieval strategy '%s'; defaulting to dense search.", active_strategy)
        query_vector = client.embed(query, settings.embedding_model)
        return dense_search(
            index,
            query_vector,
            top_k=k,
            threshold=settings.similarity_threshold,
            doc_id=doc_id,
        )

