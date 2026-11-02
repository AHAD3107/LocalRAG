"""Experimental Reciprocal Rank Fusion (RRF) Hybrid Search for LocalRAG.

Combines independent dense semantic vector search (cosine similarity) and
sparse lexical search (Okapi BM25) using reciprocal rank fusion:

    RRF_score(d) = sum_{m in {dense, bm25}} (w_m / (k + rank_m(d)))

Preserves all Chunk metadata, avoids duplicate chunks, supports document-specific
scoping (doc_id), and provides component ranking traces for debugging.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from src.ingestion.models import Chunk
from src.retrieval.bm25_search import BM25Index
from src.retrieval.search import search as dense_search
from src.storage.vector_index import VectorIndex


@dataclass
class HybridCandidateDetail:
    chunk_id: str
    rrf_score: float
    dense_rank: int | None = None
    dense_score: float | None = None
    bm25_rank: int | None = None
    bm25_score: float | None = None


@dataclass
class HybridSearchReport:
    results: list[tuple[Chunk, float]]
    details: list[HybridCandidateDetail]
    dense_count: int
    bm25_count: int


def reciprocal_rank_fusion(
    dense_results: list[tuple[Chunk, float]],
    bm25_results: list[tuple[Chunk, float]],
    rrf_k: int = 60,
    dense_weight: float = 1.0,
    bm25_weight: float = 1.0,
    top_k: int = 4,
    min_rrf_score: float = 0.0,
) -> tuple[list[tuple[Chunk, float]], list[HybridCandidateDetail]]:
    """
    Fuse two ranked lists using Reciprocal Rank Fusion.
    
    Args:
        dense_results: Ordered list of (Chunk, cosine_score) from dense retrieval.
        bm25_results: Ordered list of (Chunk, bm25_score) from sparse retrieval.
        rrf_k: Smoothing constant (default 60, Cormack et al.).
        dense_weight: Multiplier weight for dense rankings.
        bm25_weight: Multiplier weight for BM25 rankings.
        top_k: Number of fused results to return.
        min_rrf_score: Minimum combined RRF score required.
        
    Returns:
        Tuple of (ranked list of (Chunk, rrf_score), list of HybridCandidateDetail debug records).
    """
    fused_scores: dict[str, float] = {}
    chunk_map: dict[str, Chunk] = {}
    dense_ranks: dict[str, int] = {}
    dense_scores: dict[str, float] = {}
    bm25_ranks: dict[str, int] = {}
    bm25_scores: dict[str, float] = {}

    # Process Dense Rankings (1-based ranks)
    for rank, (chunk, score) in enumerate(dense_results, start=1):
        cid = chunk.chunk_id
        chunk_map[cid] = chunk
        dense_ranks[cid] = rank
        dense_scores[cid] = score
        rrf_term = dense_weight / (rrf_k + rank)
        fused_scores[cid] = fused_scores.get(cid, 0.0) + rrf_term

    # Process BM25 Rankings (1-based ranks)
    for rank, (chunk, score) in enumerate(bm25_results, start=1):
        cid = chunk.chunk_id
        chunk_map[cid] = chunk
        bm25_ranks[cid] = rank
        bm25_scores[cid] = score
        rrf_term = bm25_weight / (rrf_k + rank)
        fused_scores[cid] = fused_scores.get(cid, 0.0) + rrf_term

    if not fused_scores:
        return [], []

    # Sort descending by fused RRF score
    ranked_cids = sorted(fused_scores.keys(), key=lambda cid: -fused_scores[cid])

    results: list[tuple[Chunk, float]] = []
    details: list[HybridCandidateDetail] = []

    for cid in ranked_cids:
        score = fused_scores[cid]
        if score < min_rrf_score:
            continue

        detail = HybridCandidateDetail(
            chunk_id=cid,
            rrf_score=round(score, 6),
            dense_rank=dense_ranks.get(cid),
            dense_score=dense_scores.get(cid),
            bm25_rank=bm25_ranks.get(cid),
            bm25_score=bm25_scores.get(cid),
        )
        details.append(detail)
        results.append((chunk_map[cid], score))

        if len(results) >= top_k:
            break

    return results, details


def hybrid_search(
    index: VectorIndex,
    query: str,
    query_vector: list[float],
    bm25_index: BM25Index | None = None,
    top_k: int = 4,
    dense_threshold: float = 0.55,
    bm25_min_score: float = 0.0,
    dense_candidate_k: int = 20,
    bm25_candidate_k: int = 20,
    rrf_k: int = 60,
    dense_weight: float = 1.0,
    bm25_weight: float = 1.0,
    min_rrf_score: float = 0.0,
    doc_id: str | None = None,
) -> tuple[list[tuple[Chunk, float]], HybridSearchReport]:
    """
    Execute hybrid dense + BM25 search with Reciprocal Rank Fusion.
    
    Args:
        index: VectorIndex instance containing persistent chunk vectors.
        query: Raw query text (for BM25 lexical tokenization).
        query_vector: Dense embedding vector for the query (for cosine similarity).
        bm25_index: Optional pre-built BM25Index. If None, built dynamically from index.chunks.
        top_k: Final number of fused chunks to return.
        dense_threshold: Cosine similarity cutoff for dense candidates.
        bm25_min_score: Minimum BM25 cutoff score for sparse candidates.
        dense_candidate_k: Number of top dense candidates retrieved prior to fusion.
        bm25_candidate_k: Number of top BM25 candidates retrieved prior to fusion.
        rrf_k: RRF smoothing constant (default 60).
        dense_weight: Multiplier weight for dense rankings.
        bm25_weight: Multiplier weight for BM25 rankings.
        min_rrf_score: Minimum combined RRF score required.
        doc_id: Optional document ID filter.
        
    Returns:
        Tuple of (fused results as list of (Chunk, rrf_score), HybridSearchReport debug report).
        
    Raises:
        ValueError: If doc_id is provided but not found in index chunks.
    """
    if not index.chunks or top_k <= 0:
        if doc_id is not None:
            raise ValueError(f"Document '{doc_id}' was not found in the index.")
        empty_report = HybridSearchReport(results=[], details=[], dense_count=0, bm25_count=0)
        return [], empty_report

    # Validate doc_id up front
    if doc_id is not None:
        matching = [c for c in index.chunks if c.doc_id == doc_id]
        if not matching:
            raise ValueError(f"Document '{doc_id}' was not found in the index.")

    # 1. Retrieve dense candidates
    dense_candidates = dense_search(
        index=index,
        query_vector=query_vector,
        top_k=dense_candidate_k,
        threshold=dense_threshold,
        doc_id=doc_id,
    )

    # 2. Retrieve BM25 candidates
    if bm25_index is None:
        bm25_index = BM25Index(index.chunks)

    bm25_candidates = bm25_index.search(
        query=query,
        top_k=bm25_candidate_k,
        min_score=bm25_min_score,
        doc_id=doc_id,
    )

    # 3. Fuse rankings via RRF
    fused_results, candidate_details = reciprocal_rank_fusion(
        dense_results=dense_candidates,
        bm25_results=bm25_candidates,
        rrf_k=rrf_k,
        dense_weight=dense_weight,
        bm25_weight=bm25_weight,
        top_k=top_k,
        min_rrf_score=min_rrf_score,
    )

    report = HybridSearchReport(
        results=fused_results,
        details=candidate_details,
        dense_count=len(dense_candidates),
        bm25_count=len(bm25_candidates),
    )

    return fused_results, report

