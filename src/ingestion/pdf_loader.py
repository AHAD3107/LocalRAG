"""
PDF Loader module for LocalRAG.
Re-exports extractor functionality from pdf_extractor.py
for alignment with PDF_INGESTION_DESIGN.md naming conventions.
"""
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
