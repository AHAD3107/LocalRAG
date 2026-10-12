# Technical Design: PDF Ingestion & Text Chunking (Milestone 2)

**Status:** Proposed Architecture  
**Target Module:** `src/ingestion/`  
**Planned Implementation:** `pdf_loader.py`, `text_splitter.py`  
**Checkpoint:** Follows Foundation Checkpoint `789c21d`

---

## 1. Purpose and Scope

The purpose of the PDF ingestion subsystem is to accept binary PDF documents from the local filesystem or upload payloads, extract raw text and structural layout reliably, normalize the extracted text stream, and divide it into coherent, overlapping chunks with deterministic metadata.

### Scope Boundaries
* **In Scope (Milestone 2):**
  * Opening and validating PDF files using PyMuPDF (`fitz`).
  * Extracting page-by-page text while tracking one-based page indices.
  * Detecting corrupted, password-protected, empty, or scanned (image-only) files.
  * Normalizing whitespace while preserving paragraph structure.
  * Splitting text into configurable character-based chunks with overlap.
  * Attaching mandatory provenance metadata to every chunk (document ID, filename, page number, chunk index).
  * Automated unit and edge-case testing using on-the-fly synthetic PDFs.
* **Out of Scope (Deferred to Future Milestones):**
  * Generating vector embeddings (Milestone 3).
  * Persisting or indexing chunks into NumPy vector storage (Milestone 3).
  * Semantic retrieval or similarity ranking (Milestone 4).
  * Context-grounded LLM inference (Milestone 5).
  * FastAPI endpoints or multipart upload routes (Milestone 6).
  * Gradio frontend interface (Milestone 7).
  * Optical Character Recognition (OCR) for scanned documents (see Section 8).

---

## 2. Extraction Workflow Using PyMuPDF (`fitz`)

PyMuPDF (`fitz` version 1.24.x) is chosen for its fast C-based parsing engine, native Windows 64-bit support, low memory overhead, and resilience when encountering malformed PDF streams.

### Workflow Pipeline

```text
Input PDF Path / Bytes
         │
         ▼
[File Access & Existence Check]
         │
         ▼
[PyMuPDF fitz.open()] ──(Encrypted / Malformed?)──► [Structured Error / IngestionResult]
         │
         ▼
[Iterate Pages (0 to N-1)]
         │
         ├─► Extract text stream: page.get_text("text")
         ├─► Detect empty or raster-only page (len(text.strip()) == 0)
         ├─► Clean text (strip control chars, normalize line breaks)
         └─► Wrap into ExtractedPage (one-based page_number = index + 1)
         │
         ▼
[Chunking Engine]
         │
         ├─► Recursive split: Paragraphs (\n\n) -> Sentences -> Words -> Hard cutoff
         └─► Assemble Chunks with deterministic metadata
         │
         ▼
IngestionResult (Document + List[Chunk] + Warnings)
```

### Resource Management
Document handles must be managed via context managers or explicit `try...finally: doc.close()` blocks to prevent persistent Windows file locks (`WinError 32: The process cannot access the file because it is being used by another process`).

---

## 3. Planned Data Model

The data model will be implemented using Python standard library `dataclasses` in `src/ingestion/` to maintain zero external framework overhead.

```python
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path


class IngestionStatus(str, Enum):
    SUCCESS = "success"
    PARTIAL = "partial_success"         # Extracted with warnings (e.g. some blank pages)
    NO_TEXT = "no_extractable_text"     # Scanned or image-only document
    EMPTY = "empty_document"             # 0-page or 0-byte document
    ENCRYPTED = "encrypted_document"     # Password-protected PDF
    FAILED = "failed_corrupted"          # Unparseable / corrupted bytes


@dataclass(frozen=True)
class ExtractedPage:
    """Represents text extracted from a single PDF page."""
    page_number: int         # 1-based index (matches human reader view)
    text: str                # Normalized page text
    char_count: int
    is_empty: bool


@dataclass(frozen=True)
class Chunk:
    """Represents a coherent text fragment ready for embedding."""
    chunk_id: str            # Deterministic: f"{doc_id}_p{page_number}_c{chunk_index}"
    doc_id: str              # SHA-256 hash of source document
    source_filename: str     # Base filename (e.g. "manual.pdf")
    page_number: int         # 1-based page number where chunk originates
    chunk_index: int         # 0-based sequence number within document
    text: str                # Text content of the chunk
    char_count: int          # Character length of the chunk text


@dataclass
class Document:
    """Represents an ingested PDF document and its parsed pages."""
    doc_id: str              # SHA-256 content hash
    source_path: Path        # Absolute path on disk
    file_name: str           # e.g. "report.pdf"
    total_pages: int
    pages: list[ExtractedPage] = field(default_factory=list)


@dataclass
class IngestionResult:
    """Summary of the ingestion and chunking operation for a document."""
    doc_id: str
    source_filename: str
    status: IngestionStatus
    total_pages: int
    extracted_chunks: list[Chunk] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    error_message: str | None = None
```

---

## 4. Required Chunk Metadata

To ensure reliable, verifiable citations during retrieval and generation, every `Chunk` must preserve the following metadata attributes:

| Field | Type | Description | Purpose in RAG Pipeline |
| :--- | :--- | :--- | :--- |
| **`chunk_id`** | `str` | Unique deterministic identifier (e.g., `a1b2c3d4_p1_c0`) | Primary key for vector storage and deduplication. |
| **`doc_id`** | `str` | Hex SHA-256 digest of raw PDF file bytes | Enables document-level caching; prevents re-indexing unchanged files. |
| **`source_filename`** | `str` | Name of the source file (e.g., `annual_report.pdf`) | Displayed in citations so the user knows which document was retrieved. |
| **`page_number`** | `int` | **1-based** page number | Direct citation pointer for the user to verify in their PDF viewer. |
| **`chunk_index`** | `int` | **0-based** sequential chunk counter across the document | Preserves local reading order for potential context expansion. |
| **`text`** | `str` | Normalized text content | Input to the embedding model and LLM prompt context. |
| **`char_count`** | `int` | Character length of the chunk string | Fast diagnostics and validation against chunk size limits. |

---

## 5. Configurable Character-Based Chunk Size and Overlap

Chunking parameters are controlled via `config.py` with sensible default settings:

```python
DEFAULT_CHUNK_SIZE = 500       # Target character length per chunk
DEFAULT_CHUNK_OVERLAP = 50     # Overlap length in characters
```

### Rules & Constraints
1. **Overlap Invariant:** `0 <= chunk_overlap < chunk_size`. If invalid arguments are provided, raise a `ValueError`.
2. **Page-Bounded Chunking:** Chunks are bounded to individual pages. 
   * *Rationale:* Cross-page chunking creates ambiguous page citations (e.g. a chunk starting on page 2 and ending on page 3 makes exact citation verification difficult). Bounding chunks to page boundaries ensures every chunk has an unambiguous, 100% accurate page number.
3. **Recursive Splitting Hierarchy:**
   When splitting a page's text into chunks of target `chunk_size`:
   - Split primary separators first: double newline `\n\n` (paragraphs).
   - If a paragraph exceeds `chunk_size`, split by single newline `\n` (lines).
   - If a line exceeds `chunk_size`, split by sentence endings (`. `, `? `, `! `).
   - If a sentence exceeds `chunk_size`, split by spaces (` `).
   - As an absolute fallback for unbroken strings (e.g. long URLs or DNA sequences), split by hard character slice.
4. **Sliding Window Stitching:**
   Adjacent chunks retain a sliding window overlap of approximately `chunk_overlap` characters to preserve semantic continuity across chunk boundaries.

---

## 6. Text Normalization & Paragraph Boundary Preservation

Raw text extracted from PDFs frequently contains artifact characters, varying line breaks, and awkward hyphenation. The normalization pipeline must clean the stream without destroying paragraph boundaries.

### Normalization Rules
1. **Line Ending Standardization:** Replace Windows CRLF (`\r\n`) and legacy Mac CR (`\r`) with Unix LF (`\n`).
2. **Control Character Stripping:** Strip null bytes (`\x00`), zero-width spaces (`\u200b`, `\ufeff`), and non-printable control characters, retaining standard tabs (`\t`) and newlines (`\n`).
3. **Unicode Whitespace Normalization:** Replace non-breaking spaces (`\u00a0`) and thin spaces with standard ASCII space (` `).
4. **Paragraph Boundary Preservation:**
   - Multiple blank lines (3 or more consecutive `\n`) are collapsed to exactly two newlines (`\n\n`).
   - Trailing spaces at the end of lines are stripped (`line.rstrip()`).
   - Single newlines within continuous sentences are normalized to single spaces, *except* where preceding characters indicate lists, bullet points, or indented formatting.

---

## 7. Graceful Handling of Edge Cases & Failures

The ingestion engine must never crash or terminate the process when encountering malformed inputs. It must return structured `IngestionResult` objects with clear diagnostic statuses and warnings.

| Edge Case | Detection Mechanism | Expected System Behavior |
| :--- | :--- | :--- |
| **Missing File** | `not path.exists()` | Raises `FileNotFoundError` or returns `status=FAILED` with informative error message. |
| **Corrupted PDF** | `fitz.FileDataError` or byte corruption | Catches exception; returns `status=FAILED`, `total_pages=0`, `error_message="Corrupted or unreadable PDF"`. |
| **Password Encrypted** | `doc.is_encrypted` or `fitz.FileEncryptedError` | Catches; returns `status=ENCRYPTED`, `error_message="PDF is password protected and cannot be ingested without credentials"`. |
| **Empty File (0 bytes)** | `path.stat().st_size == 0` | Returns `status=EMPTY`, `warnings=["File is 0 bytes"]`. |
| **0 Pages in PDF** | `doc.page_count == 0` | Returns `status=EMPTY`, `warnings=["PDF contains no pages"]`. |
| **Partially Scanned PDF** | Some pages have `len(text.strip()) == 0` | Extracts available text; logs warning: `"Page X contains no extractable text"`; returns `status=PARTIAL`. |
| **Fully Scanned PDF** | All pages have `len(text.strip()) == 0` | Returns `status=NO_TEXT`, `extracted_chunks=[]`, `warnings=["No extractable text found across all N pages; document appears to be scanned/rasterized"]`. |

---

## 8. Explicit Non-Goals for Milestone 2

To maintain focus and avoid scope creep, the following components are strictly excluded from Milestone 2:

1. **Optical Character Recognition (OCR):**
   * *Rationale:* OCR requires heavy external binaries (Tesseract) or large neural models (PaddleOCR / EasyOCR), adding hundreds of megabytes of dependencies and significant latency. Milestone 2 explicitly documents lack of OCR support and gracefully rejects image-only PDFs.
2. **Ollama Embedding Calls:**
   * *Rationale:* Text chunking must be tested in total isolation without requiring the Ollama daemon or network connectivity. Embeddings belong to Milestone 3.
3. **Vector Indexing & Storage:**
   * *Rationale:* Persistence into NumPy arrays is the domain of `src/storage/` (Milestone 3).
4. **FastAPI Endpoints:**
   * *Rationale:* API routes belong to Milestone 6 and must build on already-verified core logic.
5. **LLM Generation:**
   * *Rationale:* Prompting and answer generation belong to Milestone 5.

---

## 9. Testing Strategy (Pytest & Synthetic PDFs)

All unit tests will live in `tests/test_pdf_loader.py` and `tests/test_text_splitter.py`.

### Generating Synthetic Test PDFs On the Fly
Rather than checking static binary PDF files into version control, tests will use PyMuPDF to generate synthetic in-memory or temporary PDFs dynamically via `pytest` fixtures (`tmp_path`).

```python
import fitz

def create_synthetic_pdf(path, pages_text: list[str], encrypt_pw: str = None):
    doc = fitz.open()
    for text in pages_text:
        page = doc.new_page()
        page.insert_text((50, 50), text)
    if encrypt_pw:
        doc.save(str(path), encryption=fitz.PDF_ENCRYPT_AES_256, user_pw=encrypt_pw)
    else:
        doc.save(str(path))
    doc.close()
```

### Planned Test Cases
1. `test_extract_valid_multipage_pdf`: Verifies accurate page count, text content, and 1-based page numbering across a 3-page synthetic PDF.
2. `test_chunking_size_and_overlap_invariants`: Verifies that chunks obey maximum length constraints and contain the expected overlapping text window.
3. `test_chunk_metadata_completeness`: Asserts every generated chunk possesses a non-empty `chunk_id`, valid `doc_id`, matching `source_filename`, 1-based `page_number`, and sequential `chunk_index`.
4. `test_empty_pdf_handling`: Verifies that a 0-byte file or 0-page document returns `IngestionStatus.EMPTY` without crashing.
5. `test_corrupted_pdf_handling`: Writes arbitrary random bytes to a `.pdf` file and asserts `IngestionStatus.FAILED` is cleanly returned.
6. `test_encrypted_pdf_handling`: Generates an AES-256 encrypted PDF and asserts `IngestionStatus.ENCRYPTED` is reported with appropriate warning.
7. `test_scanned_pdf_simulation`: Generates a PDF containing only vector drawings/rectangles (no text) and asserts `IngestionStatus.NO_TEXT` is reported.
8. `test_paragraph_preservation`: Verifies that text separated by `\n\n` respects paragraph boundaries during chunking.

---

## 10. Acceptance Criteria for Milestone 2 Completion

Milestone 2 will be declared complete when all of the following conditions are satisfied:

- [ ] `src/ingestion/pdf_loader.py` is implemented and extracts text cleanly with PyMuPDF.
- [ ] `src/ingestion/text_splitter.py` is implemented with configurable character size, overlap, and recursive separators.
- [ ] Document hash calculation (SHA-256) is deterministic and repeatable.
- [ ] Scanned, empty, encrypted, and corrupted PDFs produce structured status codes without unhandled exceptions.
- [ ] All chunks contain complete, validated metadata (`doc_id`, `source_filename`, 1-based `page_number`, 0-based `chunk_index`, `text`).
- [ ] Comprehensive test suites in `tests/test_pdf_loader.py` and `tests/test_text_splitter.py` pass 100% under `pytest`.
- [ ] Zero lines of unapproved external dependencies added to `requirements.txt`.
- [ ] No regression in `E:\LocalRAG\.venv` package consistency (`pip check` passes).
