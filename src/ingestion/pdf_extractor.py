"""
PDF text extraction engine using PyMuPDF (fitz).

Provides deterministic extraction, resource cleanup, text normalization,
and structured error reporting conforming to LocalRAG ingestion models.
"""
from __future__ import annotations

import logging
from pathlib import Path
import re
from typing import Sequence

import fitz  # PyMuPDF

from src.ingestion.models import (
    Document,
    ExtractedPage,
    IngestionResult,
    IngestionStatus,
    compute_sha256,
)

logger = logging.getLogger(__name__)


class PDFExtractionError(Exception):
    """Base exception for PDF text extraction failures."""

    def __init__(
        self,
        message: str,
        status: IngestionStatus = IngestionStatus.FAILED,
        source_filename: str = "",
    ) -> None:
        super().__init__(message)
        self.message = message
        self.status = status
        self.source_filename = source_filename


class EncryptedPDFError(PDFExtractionError):
    """Raised when a PDF requires a password to open."""

    def __init__(
        self,
        message: str = "PDF is password protected and cannot be extracted.",
        source_filename: str = "",
    ) -> None:
        super().__init__(
            message,
            status=IngestionStatus.ENCRYPTED,
            source_filename=source_filename,
        )


class CorruptPDFError(PDFExtractionError):
    """Raised when a PDF file is corrupted or unreadable."""

    def __init__(
        self,
        message: str = "PDF is corrupted or unreadable.",
        source_filename: str = "",
    ) -> None:
        super().__init__(
            message,
            status=IngestionStatus.FAILED,
            source_filename=source_filename,
        )


class EmptyPDFError(PDFExtractionError):
    """Raised when a PDF file is 0 bytes or contains 0 pages."""

    def __init__(
        self,
        message: str = "PDF document is empty.",
        source_filename: str = "",
    ) -> None:
        super().__init__(
            message,
            status=IngestionStatus.EMPTY,
            source_filename=source_filename,
        )


def normalize_text(text: str) -> str:
    """
    Normalize extracted PDF text stream while preserving paragraph breaks.

    Rules applied:
    1. Standardize line endings (\r\n and \r -> \n).
    2. Strip null bytes (\x00), zero-width spaces (\u200b, \ufeff).
    3. Normalize unicode whitespace variants to standard ASCII space.
    4. Strip non-printable ASCII control characters (preserving \t and \n).
    5. Strip trailing whitespace per line.
    6. Collapse 3+ consecutive newlines to \n\n (preserving paragraph breaks).
    7. Strip overall leading and trailing whitespace.
    """
    if not text:
        return ""

    # 1. Standardize line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # 2. Strip null bytes and zero-width artifacts
    text = text.replace("\x00", "").replace("\u200b", "").replace("\ufeff", "")

    # 3. Normalize unicode spaces to standard ASCII space
    for sp in ("\u00a0", "\u2009", "\u202f", "\u205f", "\u3000"):
        text = text.replace(sp, " ")

    # 4. Strip non-printable control characters (keep \t and \n)
    text = "".join(c for c in text if c in ("\t", "\n") or ord(c) >= 32)

    # 5. Strip trailing whitespace per line
    lines = [line.rstrip() for line in text.split("\n")]
    text = "\n".join(lines)

    # 6. Collapse 3 or more consecutive newlines to 2 (paragraph break)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def extract_page_text(page: fitz.Page) -> str:
    """
    Extract text from a PyMuPDF Page object, prioritizing block structure
    to preserve meaningful paragraph boundaries.
    """
    try:
        blocks = page.get_text("blocks")
        # Block format: (x0, y0, x1, y1, text, block_no, block_type)
        # block_type == 0 represents text blocks
        text_blocks = [
            b[4].strip()
            for b in blocks
            if len(b) >= 7 and b[6] == 0 and b[4].strip()
        ]
        if text_blocks:
            raw_text = "\n\n".join(text_blocks)
        else:
            raw_text = page.get_text("text")
    except Exception:
        # Fallback to standard text extraction
        raw_text = page.get_text("text")

    return normalize_text(raw_text)


def extract_pdf_result(file_path: str | Path) -> IngestionResult:
    """
    Extract text and metadata from a PDF file, returning an IngestionResult.

    Catches all standard edge cases (missing file, 0-byte, corrupt, encrypted, scanned)
    and reports them via structured IngestionStatus without raising exceptions.
    """
    path = Path(file_path)
    source_filename = path.name

    # 1. Check file existence
    if not path.exists():
        return IngestionResult(
            status=IngestionStatus.FAILED,
            source_filename=source_filename,
            error_message=f"PDF file not found: {source_filename}",
        )

    # 2. Check 0-byte file
    try:
        size = path.stat().st_size
    except OSError as err:
        return IngestionResult(
            status=IngestionStatus.FAILED,
            source_filename=source_filename,
            error_message=f"Cannot access file: {err}",
        )

    if size == 0:
        return IngestionResult(
            status=IngestionStatus.EMPTY,
            doc_id=compute_sha256(b""),
            source_filename=source_filename,
            total_pages=0,
            warnings=["File is 0 bytes"],
        )

    # 3. Read binary content for deterministic SHA-256 fingerprinting
    try:
        content = path.read_bytes()
    except OSError as err:
        return IngestionResult(
            status=IngestionStatus.FAILED,
            source_filename=source_filename,
            error_message=f"Failed to read file bytes: {err}",
        )

    doc_id = compute_sha256(content)

    # 4. Open with PyMuPDF
    doc: fitz.Document | None = None
    try:
        try:
            doc = fitz.open(stream=content, filetype="pdf")
        except (fitz.FileDataError, fitz.EmptyFileError) as err:
            return IngestionResult(
                status=IngestionStatus.FAILED,
                doc_id=doc_id,
                source_filename=source_filename,
                total_pages=0,
                error_message=f"Corrupted or unreadable PDF: {source_filename}",
            )
        except Exception as err:
            return IngestionResult(
                status=IngestionStatus.FAILED,
                doc_id=doc_id,
                source_filename=source_filename,
                total_pages=0,
                error_message=f"Failed to open PDF: {err}",
            )

        # 5. Check encryption
        if doc.is_encrypted or doc.needs_pass:
            return IngestionResult(
                status=IngestionStatus.ENCRYPTED,
                doc_id=doc_id,
                source_filename=source_filename,
                total_pages=doc.page_count,
                error_message="PDF is password protected and cannot be ingested without credentials",
            )

        # 6. Check 0-page document
        total_pages = doc.page_count
        if total_pages == 0:
            return IngestionResult(
                status=IngestionStatus.EMPTY,
                doc_id=doc_id,
                source_filename=source_filename,
                total_pages=0,
                warnings=["PDF contains no pages"],
            )

        # 7. Extract pages
        extracted_pages: list[ExtractedPage] = []
        warnings: list[str] = []
        empty_page_count = 0

        for page_idx in range(total_pages):
            page_number = page_idx + 1  # 1-based page number
            page = doc[page_idx]
            page_text = extract_page_text(page)

            extracted_page = ExtractedPage(
                page_number=page_number,
                text=page_text,
            )
            extracted_pages.append(extracted_page)

            if not extracted_page.has_text:
                empty_page_count += 1
                warnings.append(f"Page {page_number} contains no extractable text")

        # 8. Determine final ingestion status
        if empty_page_count == total_pages:
            status = IngestionStatus.NO_TEXT
            warnings = [
                f"No extractable text found across all {total_pages} pages; document appears to be scanned/rasterized"
            ]
        elif empty_page_count > 0:
            status = IngestionStatus.PARTIAL
        else:
            status = IngestionStatus.SUCCESS

        return IngestionResult(
            status=status,
            doc_id=doc_id,
            source_filename=source_filename,
            total_pages=total_pages,
            pages=extracted_pages,
            warnings=warnings,
        )

    finally:
        if doc is not None:
            doc.close()


def extract_pdf(
    file_path: str | Path,
    *,
    as_result: bool = False,
) -> Document | IngestionResult:
    """
    Extract text and metadata from a PDF file.

    By default, returns a Document containing ExtractedPage instances.
    If as_result=True, returns IngestionResult and suppresses exceptions for
    malformed, missing, empty, or encrypted documents.

    Args:
        file_path: Path to the target PDF file.
        as_result: If True, returns IngestionResult instead of Document.

    Returns:
        Document (if as_result is False) or IngestionResult (if as_result is True).

    Raises:
        FileNotFoundError: If the file does not exist.
        EmptyPDFError: If the PDF is 0 bytes or has 0 pages.
        EncryptedPDFError: If the PDF is password-protected.
        CorruptPDFError: If the PDF is corrupted or unreadable.
    """
    result = extract_pdf_result(file_path)
    if as_result:
        return result

    path = Path(file_path)
    source_filename = path.name

    if result.status == IngestionStatus.EMPTY:
        raise EmptyPDFError(
            result.warnings[0] if result.warnings else "PDF document is empty.",
            source_filename=source_filename,
        )
    if result.status == IngestionStatus.ENCRYPTED:
        raise EncryptedPDFError(
            result.error_message or "PDF is password protected and cannot be extracted.",
            source_filename=source_filename,
        )
    if result.status == IngestionStatus.FAILED:
        if "not found" in (result.error_message or "").lower():
            raise FileNotFoundError(
                result.error_message or f"PDF file not found: {source_filename}"
            )
        raise CorruptPDFError(
            result.error_message or "PDF is corrupted or unreadable.",
            source_filename=source_filename,
        )

    return Document(
        doc_id=result.doc_id or "",
        source_filename=result.source_filename,
        pages=result.pages,
        source_path=path,
    )


def extract_pdf_document(file_path: str | Path) -> Document:
    """
    Convenience alias returning Document directly.
    Equivalent to extract_pdf(file_path, as_result=False).
    """
    doc = extract_pdf(file_path, as_result=False)
    assert isinstance(doc, Document)
    return doc
