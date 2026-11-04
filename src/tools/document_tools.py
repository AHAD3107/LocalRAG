"""Document management and search tools for LocalRAG."""
from __future__ import annotations

import logging
from typing import Any

from config import settings
from src.embeddings.ollama import OllamaClient
from src.retrieval.pipeline import retrieve_context
from src.storage.vector_index import VectorIndex, IndexError as LocalIndexError

logger = logging.getLogger(__name__)


def _get_index() -> VectorIndex:
    """Load the current persistent vector index."""
    try:
        return VectorIndex.load(
            settings.index_dir,
            settings.embedding_model,
            settings.chunk_size,
            settings.chunk_overlap,
        )
    except Exception as exc:
        logger.error("Failed to load vector index: %s", exc)
        raise RuntimeError(f"Vector index unavailable: {exc}") from exc


def list_documents() -> list[dict[str, Any]]:
    """
    List all indexed documents available in LocalRAG.

    Returns:
        List of document dictionaries containing document_id, filename, chunk_count,
        page_count, and index_status.
    """
    index = _get_index()
    docs: dict[str, dict[str, Any]] = {}
    for chunk in index.chunks:
        if chunk.doc_id not in docs:
            docs[chunk.doc_id] = {
                "document_id": chunk.doc_id,
                "filename": chunk.source_filename,
                "chunk_count": 0,
                "pages": set(),
                "index_status": "indexed",
            }
        docs[chunk.doc_id]["chunk_count"] += 1
        docs[chunk.doc_id]["pages"].add(chunk.page_number)

    result = []
    for doc in docs.values():
        pages = sorted(list(doc.pop("pages")))
        doc["page_count"] = len(pages)
        doc["page_range"] = [min(pages), max(pages)] if pages else [0, 0]
        result.append(doc)
    return result


def get_document_info(document_id: str) -> dict[str, Any]:
    """
    Retrieve comprehensive metadata about a specific indexed document.

    Args:
        document_id: The unique document identifier or filename.

    Returns:
        Dictionary containing document metadata, page numbers, and chunk statistics.
    """
    if not document_id or not isinstance(document_id, str):
        return {"error": "document_id must be a non-empty string"}

    index = _get_index()
    doc_id_clean = document_id.strip()

    # Match by exact doc_id or by filename prefix/match
    matching_chunks = [
        c for c in index.chunks
        if c.doc_id == doc_id_clean or c.source_filename.lower() == doc_id_clean.lower()
    ]

    if not matching_chunks:
        # Check partial match on filename for convenience
        matching_chunks = [
            c for c in index.chunks
            if doc_id_clean.lower() in c.source_filename.lower()
        ]

    if not matching_chunks and doc_id_clean.lower() in (
        "document_id",
        "document",
        "uploaded_document",
        "the document",
        "this document",
        "the uploaded document",
        "default",
    ):
        # Fallback to the first available document if a literal placeholder was passed
        if index.chunks:
            target_first_id = index.chunks[0].doc_id
            matching_chunks = [c for c in index.chunks if c.doc_id == target_first_id]

    if not matching_chunks:
        return {
            "error": f"Document '{document_id}' was not found in the index.",
            "document_id": document_id,
            "status": "not_found",
        }

    first_chunk = matching_chunks[0]
    pages = sorted(list({c.page_number for c in matching_chunks}))
    return {
        "document_id": first_chunk.doc_id,
        "filename": first_chunk.source_filename,
        "chunk_count": len(matching_chunks),
        "page_count": len(pages),
        "pages_available": pages,
        "first_page": min(pages) if pages else 1,
        "last_page": max(pages) if pages else 1,
        "index_status": "indexed",
        "embedding_model": index.embedding_model,
    }


def search_documents(
    query: str,
    document_id: str | None = None,
    top_k: int | None = None,
    retrieval_strategy: str | None = None,
) -> dict[str, Any]:
    """
    Search indexed documents using the configured LocalRAG retrieval pipeline.

    Args:
        query: User search query string (max 2000 chars).
        document_id: Optional document ID or filename to scope retrieval.
        top_k: Number of ranked passages to return (1 to 20, default 4).
        retrieval_strategy: Strategy override ('hybrid_rerank', 'bm25_rerank', 'hybrid', 'bm25', 'dense').

    Returns:
        Dictionary with query, strategy_used, result_count, and ranked passages list.
    """
    if not query or not isinstance(query, str) or not query.strip():
        return {"error": "Query must contain non-whitespace text.", "results": []}

    clean_query = query.strip()[:2000]
    k = max(1, min(top_k or settings.top_k, 20))
    strat = (retrieval_strategy or settings.retrieval_strategy).lower().strip()

    valid_strats = {"hybrid_rerank", "bm25_rerank", "hybrid", "bm25", "dense"}
    if strat not in valid_strats:
        strat = settings.retrieval_strategy

    index = _get_index()
    if not index.chunks:
        return {
            "query": clean_query,
            "strategy_used": strat,
            "result_count": 0,
            "results": [],
            "message": "Index is empty.",
        }

    # Resolve document_id if provided as a filename
    target_doc_id = None
    if document_id:
        doc_info = get_document_info(document_id)
        if "error" in doc_info:
            return {
                "error": f"Scoped document '{document_id}' not found.",
                "results": [],
            }
        target_doc_id = doc_info["document_id"]

    client = OllamaClient(settings.ollama_base_url, settings.request_timeout_seconds)
    try:
        ranked_chunks = retrieve_context(
            query=clean_query,
            index=index,
            client=client,
            top_k=k,
            doc_id=target_doc_id,
            strategy=strat,
        )
    except Exception as exc:
        logger.warning("Retrieval failed for '%s': %s", clean_query[:50], exc)
        return {
            "error": f"Retrieval error: {exc}",
            "query": clean_query,
            "results": [],
        }
    finally:
        if hasattr(client, "close"):
            client.close()

    results = []
    for chunk, score in ranked_chunks:
        results.append({
            "document_id": chunk.doc_id,
            "filename": chunk.source_filename,
            "page_number": chunk.page_number,
            "chunk_index": chunk.chunk_index,
            "score": round(float(score), 4),
            "text": chunk.text,
            "citation": f"{chunk.source_filename} - Page {chunk.page_number}",
        })

    return {
        "query": clean_query,
        "strategy_used": strat,
        "scoped_document_id": target_doc_id,
        "result_count": len(results),
        "results": results,
    }


def retrieve_passages(
    document_id: str,
    page_number: int | None = None,
    page_range: list[int] | None = None,
    limit: int = 5,
) -> dict[str, Any]:
    """
    Retrieve deterministic passages directly from a document by page number or range.

    Args:
        document_id: Target document ID or filename.
        page_number: Specific page number to retrieve.
        page_range: [start_page, end_page] inclusive.
        limit: Maximum number of passages to return (1 to 20).

    Returns:
        Dictionary containing matching passages and source citations.
    """
    if not document_id:
        return {"error": "document_id is required", "passages": []}

    doc_info = get_document_info(document_id)
    if "error" in doc_info:
        return {"error": doc_info["error"], "passages": []}

    target_id = doc_info["document_id"]
    filename = doc_info["filename"]
    max_limit = max(1, min(limit, 20))

    index = _get_index()
    chunks = [c for c in index.chunks if c.doc_id == target_id]

    if page_number is not None:
        target_pages = {int(page_number)}
    elif page_range and len(page_range) >= 2:
        target_pages = set(range(int(page_range[0]), int(page_range[1]) + 1))
    else:
        target_pages = None

    filtered = []
    for c in chunks:
        if target_pages is None or c.page_number in target_pages:
            filtered.append(c)

    filtered.sort(key=lambda c: (c.page_number, c.chunk_index))
    capped = filtered[:max_limit]

    passages = [
        {
            "document_id": c.doc_id,
            "filename": filename,
            "page_number": c.page_number,
            "chunk_index": c.chunk_index,
            "text": c.text,
            "citation": f"{filename} - Page {c.page_number}",
        }
        for c in capped
    ]

    return {
        "document_id": target_id,
        "filename": filename,
        "requested_page": page_number,
        "passages_count": len(passages),
        "total_available": len(filtered),
        "passages": passages,
    }
