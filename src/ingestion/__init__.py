"""
Document ingestion subsystem for LocalRAG.
"""
from src.ingestion.models import (
    Chunk,
    Document,
    ExtractedPage,
    IngestionResult,
    IngestionStatus,
    compute_chunk_id,
    compute_sha256,
)
from src.ingestion.pdf_extractor import (
    CorruptPDFError,
    EmptyPDFError,
    EncryptedPDFError,
    PDFExtractionError,
    extract_page_text,
    extract_pdf,
    extract_pdf_document,
    extract_pdf_result,
    normalize_text,
)

__all__ = [
    "Chunk",
    "Document",
    "ExtractedPage",
    "IngestionResult",
    "IngestionStatus",
    "compute_chunk_id",
    "compute_sha256",
    "CorruptPDFError",
    "EmptyPDFError",
    "EncryptedPDFError",
    "PDFExtractionError",
    "extract_page_text",
    "extract_pdf",
    "extract_pdf_document",
    "extract_pdf_result",
    "normalize_text",
]
