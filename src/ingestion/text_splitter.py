"""Deterministic page-bounded text chunking."""
from __future__ import annotations

from src.ingestion.models import Chunk, Document, compute_chunk_id


def chunk_document(document: Document, chunk_size: int = 500, overlap: int = 50) -> list[Chunk]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must satisfy 0 <= overlap < chunk_size")

    chunks: list[Chunk] = []
    for page in document.pages:
        if not page.text.strip():
            continue
        text = page.text.strip()
        start = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            piece = text[start:end].strip()
            if piece:
                chunk_index = len(chunks)
                chunks.append(
                    Chunk(
                        chunk_id=compute_chunk_id(document.doc_id, page.page_number, chunk_index),
                        doc_id=document.doc_id,
                        source_filename=document.source_filename,
                        page_number=page.page_number,
                        chunk_index=chunk_index,
                        text=piece,
                    )
                )
            if end >= len(text):
                break
            next_start = end - overlap
            start = next_start if next_start > start else end
    return chunks
