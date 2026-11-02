"""Cosine-similarity retrieval."""
from __future__ import annotations

import numpy as np

from src.ingestion.models import Chunk
from src.storage.vector_index import VectorIndex


def search(
    index: VectorIndex,
    query_vector: list[float],
    top_k: int = 4,
    threshold: float = 0.35,
    doc_id: str | None = None,
) -> list[tuple[Chunk, float]]:
    if top_k <= 0 or not index.chunks:
        if doc_id is not None:
            raise ValueError(f"Document '{doc_id}' was not found in the index.")
        return []

    if doc_id is not None:
        matching_indices = [i for i, c in enumerate(index.chunks) if c.doc_id == doc_id]
        if not matching_indices:
            raise ValueError(f"Document '{doc_id}' was not found in the index.")
        candidate_chunks = [index.chunks[i] for i in matching_indices]
        candidate_vectors = index.vectors[matching_indices]
    else:
        candidate_chunks = index.chunks
        candidate_vectors = index.vectors

    query = np.asarray(query_vector, dtype=np.float32)
    if query.ndim != 1 or candidate_vectors.ndim != 2 or query.shape[0] != candidate_vectors.shape[1]:
        raise ValueError("query embedding dimension does not match index")
    norm = np.linalg.norm(query)
    if norm == 0:
        return []
    scores = candidate_vectors @ (query / norm)
    order = np.argsort(-scores)[:top_k]
    return [(candidate_chunks[int(i)], float(scores[i])) for i in order if scores[i] >= threshold]
