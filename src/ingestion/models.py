"""
Core data models and identifiers for the document ingestion pipeline.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import hashlib
from pathlib import Path


def compute_sha256(content: bytes) -> str:
    """
    Compute a deterministic hex-encoded SHA-256 fingerprint from raw bytes.

    Args:
        content: Binary content of a document.

    Returns:
        64-character lowercase hexadecimal SHA-256 string.

    Raises:
        TypeError: If content is not bytes or bytearray.
    """
    if not isinstance(content, (bytes, bytearray)):
        raise TypeError(f"content must be bytes or bytearray, got {type(content).__name__}")
    return hashlib.sha256(content).hexdigest()


def compute_chunk_id(doc_id: str, page_number: int, chunk_index: int) -> str:
    """
    Generate a deterministic, reproducible chunk identifier.

    The format is: `{doc_id}_p{page_number}_c{chunk_index}`.

    Args:
        doc_id: Stable document SHA-256 fingerprint.
        page_number: 1-based page number where the chunk originated.
        chunk_index: 0-based sequence index of the chunk within the document.

    Returns:
        Deterministic chunk identifier string.

    Raises:
        ValueError: If doc_id is empty, page_number < 1, or chunk_index < 0.
    """
    if not isinstance(doc_id, str) or not doc_id.strip():
        raise ValueError("doc_id cannot be empty when generating chunk_id")
    if not isinstance(page_number, int) or isinstance(page_number, bool) or page_number < 1:
        raise ValueError(f"page_number must be an integer >= 1, got {page_number}")
    if not isinstance(chunk_index, int) or isinstance(chunk_index, bool) or chunk_index < 0:
        raise ValueError(f"chunk_index must be an integer >= 0, got {chunk_index}")
    return f"{doc_id.strip()}_p{page_number}_c{chunk_index}"


class IngestionStatus(str, Enum):
    """
    Standard outcome statuses for document ingestion operations.

    SUCCESS:
        All document pages were processed cleanly and produced usable chunks.
    PARTIAL:
        The document produced usable chunks, but one or more pages contained
        no extractable text (e.g. blank separator pages or embedded images).
        The document is still usable for question answering.
    NO_TEXT:
        The document opened successfully, but zero extractable text was found
        across all pages (e.g. scanned/rasterized PDFs lacking an OCR text layer).
    EMPTY:
        The document is empty (0 pages or 0 bytes).
    ENCRYPTED:
        The document is password-protected and cannot be read without credentials.
    FAILED:
        The document is corrupted, malformed, or failed with an unrecoverable error.
    """
    SUCCESS = "success"
    PARTIAL = "partial_success"
    NO_TEXT = "no_extractable_text"
    EMPTY = "empty_document"
    ENCRYPTED = "encrypted_document"
    FAILED = "failed_corrupted"


@dataclass(frozen=True)
class ExtractedPage:
    """
    Represents raw text extracted from a single PDF page.

    Attributes:
        page_number: 1-based page index (matches human reader view).
        text: Extracted page text content.
        has_text: True if text contains non-whitespace characters.
        text_length: Total character count of extracted text.
    """
    page_number: int
    text: str
    has_text: bool = field(init=False)
    text_length: int = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.page_number, int) or isinstance(self.page_number, bool):
            raise TypeError(f"page_number must be an int, got {type(self.page_number).__name__}")
        if self.page_number < 1:
            raise ValueError(f"page_number must be >= 1 (1-based), got {self.page_number}")
        if not isinstance(self.text, str):
            raise TypeError(f"text must be a str, got {type(self.text).__name__}")

        # Compute derived invariants deterministically
        has_usable_text = bool(self.text.strip())
        length = len(self.text)
        object.__setattr__(self, "has_text", has_usable_text)
        object.__setattr__(self, "text_length", length)

    @property
    def is_empty(self) -> bool:
        """Alias matching PDF_INGESTION_DESIGN.md semantics."""
        return not self.has_text

    @property
    def char_count(self) -> int:
        """Alias matching PDF_INGESTION_DESIGN.md semantics."""
        return self.text_length


@dataclass(frozen=True)
class Chunk:
    """
    Represents a coherent, page-bounded text chunk ready for vectorization.

    Attributes:
        chunk_id: Deterministic identifier derived from doc_id, page_number, and chunk_index.
        doc_id: SHA-256 fingerprint of the source document.
        source_filename: Name of the originating file (for citations).
        page_number: 1-based page number where this chunk resides.
        chunk_index: 0-based sequential counter within the document.
        text: Non-empty chunk text content.
        char_count: Character count of the chunk text.
    """
    chunk_id: str
    doc_id: str
    source_filename: str
    page_number: int
    chunk_index: int
    text: str
    char_count: int = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.page_number, int) or isinstance(self.page_number, bool):
            raise TypeError(f"page_number must be an int, got {type(self.page_number).__name__}")
        if self.page_number < 1:
            raise ValueError(f"page_number must be >= 1, got {self.page_number}")

        if not isinstance(self.chunk_index, int) or isinstance(self.chunk_index, bool):
            raise TypeError(f"chunk_index must be an int, got {type(self.chunk_index).__name__}")
        if self.chunk_index < 0:
            raise ValueError(f"chunk_index must be >= 0, got {self.chunk_index}")

        if not isinstance(self.text, str) or not self.text.strip():
            raise ValueError("chunk text must be a non-empty, non-whitespace string")

        if not isinstance(self.doc_id, str) or not self.doc_id.strip():
            raise ValueError("doc_id must be a non-empty string")

        if not isinstance(self.source_filename, str) or not self.source_filename.strip():
            raise ValueError("source_filename must be a non-empty string")

        if not isinstance(self.chunk_id, str) or not self.chunk_id.strip():
            raise ValueError("chunk_id must be a non-empty string")

        object.__setattr__(self, "char_count", len(self.text))


@dataclass
class Document:
    """
    Represents a parsed source document and its extracted pages.

    Attributes:
        doc_id: SHA-256 fingerprint of the source document bytes.
        source_filename: Source file name (e.g. 'manual.pdf').
        pages: Ordered sequence of ExtractedPage records.
        source_path: Optional local filesystem path where the file was located.
    """
    doc_id: str
    source_filename: str
    pages: list[ExtractedPage] = field(default_factory=list)
    source_path: Path | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.doc_id, str) or not self.doc_id.strip():
            raise ValueError("doc_id cannot be empty")
        if not isinstance(self.source_filename, str) or not self.source_filename.strip():
            raise ValueError("source_filename cannot be empty")

        # Validate page ordering and integrity
        if self.pages:
            seen_pages: set[int] = set()
            for idx, p in enumerate(self.pages):
                if not isinstance(p, ExtractedPage):
                    raise TypeError(f"pages must contain ExtractedPage instances, got {type(p).__name__}")
                if p.page_number in seen_pages:
                    raise ValueError(f"Duplicate page_number {p.page_number} found in document pages")
                seen_pages.add(p.page_number)
                if idx > 0 and p.page_number <= self.pages[idx - 1].page_number:
                    raise ValueError(
                        f"Pages must be ordered by ascending page_number. Page {p.page_number} followed page {self.pages[idx - 1].page_number}"
                    )

    @property
    def total_pages(self) -> int:
        """Total number of extracted pages in this document."""
        return len(self.pages)

    @property
    def file_name(self) -> str:
        """Alias matching PDF_INGESTION_DESIGN.md semantics."""
        return self.source_filename


@dataclass
class IngestionResult:
    """
    Structured outcome of an ingestion operation.

    Attributes:
        status: Standard IngestionStatus.
        doc_id: SHA-256 document fingerprint if available, else None.
        source_filename: Source filename.
        total_pages: Total number of pages in the document.
        pages: List of extracted ExtractedPage records.
        chunks: List of generated Chunk records.
        warnings: Non-fatal diagnostic messages.
        error_message: Fatal error description if status is FAILED or ENCRYPTED.
    """
    status: IngestionStatus
    doc_id: str | None = None
    source_filename: str = ""
    total_pages: int = 0
    pages: list[ExtractedPage] = field(default_factory=list)
    chunks: list[Chunk] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    error_message: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.status, IngestionStatus):
            raise TypeError(f"status must be an IngestionStatus enum member, got {type(self.status).__name__}")

    @property
    def is_usable(self) -> bool:
        """
        True if ingestion produced usable content (SUCCESS or PARTIAL with chunks).
        A document with some empty pages is still usable if other pages produced chunks.
        """
        return self.status in (IngestionStatus.SUCCESS, IngestionStatus.PARTIAL) and len(self.chunks) > 0

    @property
    def extracted_chunks(self) -> list[Chunk]:
        """Alias matching PDF_INGESTION_DESIGN.md naming."""
        return self.chunks
