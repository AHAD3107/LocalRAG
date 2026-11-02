"""Experimental Cross-Encoder Reranker for LocalRAG.

Evaluates second-stage cross-encoder reranking on top of dense, BM25, or hybrid candidates.
Computes fine-grained query-document attention scores using a transformer cross-encoder.

Preserves all Chunk metadata, provides robust error handling with graceful fallback,
and supports device selection (CUDA / CPU) with calibrated score thresholds.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any, Sequence

from src.ingestion.models import Chunk

logger = logging.getLogger(__name__)

DEFAULT_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


@dataclass
class RerankCandidateDetail:
    """Detailed trace for a reranked candidate."""
    chunk_id: str
    original_rank: int
    original_score: float
    rerank_score: float
    sigmoid_score: float


@dataclass
class RerankReport:
    """Full execution report containing reranked results and diagnostic details."""
    results: list[tuple[Chunk, float]]
    details: list[RerankCandidateDetail]
    model_name: str
    device: str
    total_candidates: int
    returned_candidates: int


def _sigmoid(x: float) -> float:
    """Numerically stable sigmoid function."""
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    else:
        z = math.exp(x)
        return z / (1.0 + z)


class CrossEncoderReranker:
    """
    Second-stage cross-encoder reranker.

    Takes first-stage candidate chunks (from dense, BM25, or hybrid search)
    and scores (query, chunk_text) pairs jointly using full self-attention.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_RERANKER_MODEL,
        device: str | None = None,
        lazy_load: bool = True,
    ) -> None:
        self.model_name = model_name
        self._requested_device = device
        self._resolved_device: str | None = None
        self._model: Any = None

        if not lazy_load:
            self._ensure_loaded()

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def device(self) -> str:
        if self._resolved_device is not None:
            return self._resolved_device
        if self._requested_device is not None:
            return self._requested_device
        return "cpu"

    def _resolve_device(self) -> str:
        if self._requested_device:
            return self._requested_device
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return

        self._resolved_device = self._resolve_device()
        logger.info(
            "Loading CrossEncoder model '%s' on device '%s'...",
            self.model_name,
            self._resolved_device,
        )
        try:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(
                self.model_name,
                device=self._resolved_device,
            )
            logger.info("Successfully loaded CrossEncoder '%s'", self.model_name)
        except Exception as exc:
            logger.error("Failed to load CrossEncoder model '%s': %s", self.model_name, exc)
            raise

    def score_pairs(self, pairs: Sequence[tuple[str, str]]) -> list[float]:
        """
        Score a sequence of (query, document_text) pairs.

        Returns raw model logits as floats.
        """
        if not pairs:
            return []

        self._ensure_loaded()
        try:
            scores = self._model.predict(
                list(pairs),
                convert_to_numpy=True,
                show_progress_bar=False,
            )
            return [float(s) for s in scores]
        except Exception as exc:
            logger.error("Error during CrossEncoder predict: %s", exc)
            raise

    def rerank(
        self,
        query: str,
        candidates: Sequence[tuple[Chunk, float]],
        top_k: int = 4,
        score_threshold: float | None = None,
    ) -> list[tuple[Chunk, float]]:
        """
        Rerank a sequence of (Chunk, first_stage_score) tuples.

        Args:
            query: The retrieval query string.
            candidates: Sequence of (Chunk, original_score) from first-stage retrieval.
            top_k: Maximum number of reranked candidates to return.
            score_threshold: Optional cutoff for raw reranker logit score.
                             Candidates with score < score_threshold are dropped.

        Returns:
            List of (Chunk, rerank_score) sorted by rerank_score descending.
            If candidates is empty, returns empty list.
            If model inference fails, logs error and falls back to original candidates.
        """
        report = self.rerank_with_report(
            query=query,
            candidates=candidates,
            top_k=top_k,
            score_threshold=score_threshold,
        )
        return report.results

    def rerank_with_report(
        self,
        query: str,
        candidates: Sequence[tuple[Chunk, float]],
        top_k: int = 4,
        score_threshold: float | None = None,
    ) -> RerankReport:
        """
        Rerank candidates and return structured diagnostic report.
        """
        if not candidates or top_k <= 0:
            return RerankReport(
                results=[],
                details=[],
                model_name=self.model_name,
                device=self.device,
                total_candidates=len(candidates) if candidates else 0,
                returned_candidates=0,
            )

        pairs = [(query, chunk.text) for chunk, _ in candidates]

        try:
            raw_scores = self.score_pairs(pairs)
        except Exception as exc:
            logger.warning(
                "Reranker failed during scoring (%s); falling back to first-stage ranking.",
                exc,
            )
            fallback_results = list(candidates)[:top_k]
            fallback_details = [
                RerankCandidateDetail(
                    chunk_id=chunk.chunk_id,
                    original_rank=i + 1,
                    original_score=score,
                    rerank_score=score,
                    sigmoid_score=_sigmoid(score),
                )
                for i, (chunk, score) in enumerate(fallback_results)
            ]
            return RerankReport(
                results=fallback_results,
                details=fallback_details,
                model_name=self.model_name,
                device=self.device,
                total_candidates=len(candidates),
                returned_candidates=len(fallback_results),
            )

        scored_items: list[tuple[Chunk, float, int, float, float]] = []
        for rank_0, ((chunk, orig_score), rerank_score) in enumerate(zip(candidates, raw_scores)):
            sig_score = _sigmoid(rerank_score)
            scored_items.append((chunk, rerank_score, rank_0 + 1, orig_score, sig_score))

        # Sort descending by rerank_score
        scored_items.sort(key=lambda item: item[1], reverse=True)

        # Apply threshold filtering if specified
        if score_threshold is not None:
            scored_items = [item for item in scored_items if item[1] >= score_threshold]

        final_items = scored_items[:top_k]

        results = [(chunk, score) for chunk, score, _, _, _ in final_items]
        details = [
            RerankCandidateDetail(
                chunk_id=chunk.chunk_id,
                original_rank=orig_rank,
                original_score=orig_score,
                rerank_score=score,
                sigmoid_score=sig_score,
            )
            for chunk, score, orig_rank, orig_score, sig_score in final_items
        ]

        return RerankReport(
            results=results,
            details=details,
            model_name=self.model_name,
            device=self.device,
            total_candidates=len(candidates),
            returned_candidates=len(results),
        )

