"""
Unit and integration tests for the PyMuPDF PDF text extraction engine.
Uses temporary synthetic PDFs created dynamically with PyMuPDF.
"""
from pathlib import Path

import fitz
import pytest

from src.ingestion.models import (
    Document,
    ExtractedPage,
    IngestionResult,
    IngestionStatus,
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


def create_synthetic_pdf(
    path: Path,
    pages_text: list[str | None],
    encrypt_pw: str | None = None,
) -> Path:
    """
    Helper to generate synthetic PDF files dynamically for testing.
    If a page text is None, the page is left empty (or only shapes added).
    """
    doc = fitz.open()
    for text in pages_text:
        page = doc.new_page()
        if text is not None:
            # Insert text into page
            page.insert_textbox(fitz.Rect(50, 50, 500, 700), text)
        else:
            # Draw a shape to simulate a scanned image / drawing with no text
            page.draw_rect(fitz.Rect(50, 50, 300, 300), color=(0, 0, 1), fill=(0, 1, 0))

    if encrypt_pw:
        doc.save(str(path), encryption=fitz.PDF_ENCRYPT_AES_256, user_pw=encrypt_pw)
    else:
        doc.save(str(path))
    doc.close()
    return path


class TestPDFExtraction:
    def test_extract_valid_multipage_pdf(self, tmp_path: Path):
        pdf_path = tmp_path / "valid_multipage.pdf"
        pages = [
            "Introduction to LocalRAG\nThis is the first page of the document.",
            "Architecture Overview\nThis is the second page explaining components.",
            "Evaluation and Results\nThis is the third page presenting benchmarks.",
        ]
        create_synthetic_pdf(pdf_path, pages)

        # 1. Test extract_pdf (returns Document)
        doc = extract_pdf(pdf_path)
        assert isinstance(doc, Document)
        assert doc.source_filename == "valid_multipage.pdf"
        assert doc.total_pages == 3
        assert len(doc.pages) == 3

        # Verify 1-based page numbers
        for idx, page in enumerate(doc.pages):
            assert isinstance(page, ExtractedPage)
            assert page.page_number == idx + 1
            assert page.has_text is True
            assert page.is_empty is False

        assert "Introduction to LocalRAG" in doc.pages[0].text
        assert "Architecture Overview" in doc.pages[1].text
        assert "Evaluation and Results" in doc.pages[2].text

        # Verify content SHA-256 fingerprint
        expected_hash = compute_sha256(pdf_path.read_bytes())
        assert doc.doc_id == expected_hash

        # 2. Test extract_pdf_result (returns IngestionResult)
        result = extract_pdf_result(pdf_path)
        assert isinstance(result, IngestionResult)
        assert result.status == IngestionStatus.SUCCESS
        assert result.total_pages == 3
        assert len(result.pages) == 3
        assert len(result.warnings) == 0
        assert result.error_message is None
        assert result.doc_id == expected_hash

    def test_extract_pdf_document_helper(self, tmp_path: Path):
        pdf_path = tmp_path / "helper_test.pdf"
        create_synthetic_pdf(pdf_path, ["Testing convenience helper function."])

        doc = extract_pdf_document(pdf_path)
        assert isinstance(doc, Document)
        assert doc.total_pages == 1
        assert "Testing convenience helper" in doc.pages[0].text

    def test_page_boundaries_and_paragraphs(self, tmp_path: Path):
        pdf_path = tmp_path / "paragraphs.pdf"
        doc = fitz.open()
        page = doc.new_page()
        # Insert two separate text blocks simulating distinct paragraphs
        page.insert_textbox(fitz.Rect(50, 50, 500, 150), "Paragraph One line 1.\nParagraph One line 2.")
        page.insert_textbox(fitz.Rect(50, 200, 500, 300), "Paragraph Two line 1.\nParagraph Two line 2.")
        doc.save(str(pdf_path))
        doc.close()

        extracted_doc = extract_pdf(pdf_path)
        page_text = extracted_doc.pages[0].text
        assert "Paragraph One line 1." in page_text
        assert "Paragraph Two line 1." in page_text
        assert "\n\n" in page_text  # Meaningful paragraph boundary preserved

    def test_text_normalization(self):
        sample = (
            "Header text\x00\r\n"
            "Line with unicode\u00a0space and zero\u200bwidth.\r\n"
            "Line with trailing spaces   \n\n\n\n"
            "New paragraph after multiple blanks."
        )
        cleaned = normalize_text(sample)
        assert "\x00" not in cleaned
        assert "\u200b" not in cleaned
        assert "\u00a0" not in cleaned
        assert "\r" not in cleaned
        assert "Line with trailing spaces\n\nNew paragraph" in cleaned
        assert not cleaned.endswith(" ")

    def test_empty_pages_handling_partial_status(self, tmp_path: Path):
        pdf_path = tmp_path / "partially_empty.pdf"
        pages = [
            "Valid page 1 content.",
            None,  # Blank / shape-only page
            "Valid page 3 content.",
        ]
        create_synthetic_pdf(pdf_path, pages)

        # Document extraction should still succeed with pages marked has_text=False
        doc = extract_pdf(pdf_path)
        assert doc.total_pages == 3
        assert doc.pages[0].has_text is True
        assert doc.pages[1].has_text is False
        assert doc.pages[1].is_empty is True
        assert doc.pages[2].has_text is True

        # Result extraction should report PARTIAL status and a warning
        result = extract_pdf_result(pdf_path)
        assert result.status == IngestionStatus.PARTIAL
        assert len(result.warnings) == 1
        assert "Page 2 contains no extractable text" in result.warnings[0]

    def test_fully_scanned_pdf_simulation_no_text(self, tmp_path: Path):
        pdf_path = tmp_path / "scanned_simulation.pdf"
        # Two pages with zero text
        create_synthetic_pdf(pdf_path, [None, None])

        doc = extract_pdf(pdf_path)
        assert doc.total_pages == 2
        assert all(not p.has_text for p in doc.pages)

        result = extract_pdf_result(pdf_path)
        assert result.status == IngestionStatus.NO_TEXT
        assert any("scanned/rasterized" in w for w in result.warnings)

    def test_missing_file_handling(self, tmp_path: Path):
        missing_path = tmp_path / "does_not_exist.pdf"

        with pytest.raises(FileNotFoundError, match="(?i)PDF file not found"):
            extract_pdf(missing_path)

        result = extract_pdf_result(missing_path)
        assert result.status == IngestionStatus.FAILED
        assert "not found" in (result.error_message or "").lower()

    def test_zero_byte_file_handling(self, tmp_path: Path):
        empty_path = tmp_path / "empty_file.pdf"
        empty_path.write_bytes(b"")

        with pytest.raises(EmptyPDFError, match="(?i)0 bytes"):
            extract_pdf(empty_path)

        result = extract_pdf_result(empty_path)
        assert result.status == IngestionStatus.EMPTY
        assert any("0 bytes" in w for w in result.warnings)

    def test_corrupt_malformed_pdf_handling(self, tmp_path: Path):
        corrupt_path = tmp_path / "corrupted.pdf"
        corrupt_path.write_bytes(b"%PDF-1.4 RANDOM_GARBAGE_BYTES_CORRUPTED_STREAM_1234567890")

        with pytest.raises(CorruptPDFError, match="(?i)corrupted or unreadable"):
            extract_pdf(corrupt_path)

        result = extract_pdf_result(corrupt_path)
        assert result.status == IngestionStatus.FAILED
        assert result.error_message is not None
        assert "corrupted or unreadable" in result.error_message.lower()

    def test_encrypted_pdf_handling(self, tmp_path: Path):
        encrypted_path = tmp_path / "encrypted_doc.pdf"
        create_synthetic_pdf(encrypted_path, ["Top secret confidential"], encrypt_pw="pass123")

        with pytest.raises(EncryptedPDFError, match="(?i)password protected"):
            extract_pdf(encrypted_path)

        result = extract_pdf_result(encrypted_path)
        assert result.status == IngestionStatus.ENCRYPTED
        assert result.error_message is not None
        assert "password protected" in result.error_message.lower()

    def test_resource_cleanup_no_windows_file_lock(self, tmp_path: Path):
        """Verify that extracting a PDF closes all file handles so the file can be immediately removed."""
        pdf_path = tmp_path / "lock_test.pdf"
        create_synthetic_pdf(pdf_path, ["Checking Windows file lock closure."])

        # Extract document
        doc = extract_pdf(pdf_path)
        assert doc.total_pages == 1

        # Must not raise PermissionError / WinError 32
        pdf_path.unlink()
        assert not pdf_path.exists()

    def test_pdf_loader_alias_compatibility(self, tmp_path: Path):
        """Verify that the alias module src.ingestion.pdf_loader works identically."""
        from src.ingestion.pdf_loader import extract_pdf as loader_extract_pdf

        pdf_path = tmp_path / "loader_alias.pdf"
        create_synthetic_pdf(pdf_path, ["Page from loader alias."])

        doc = loader_extract_pdf(pdf_path)
        assert isinstance(doc, Document)
        assert doc.total_pages == 1
        assert "Page from loader alias" in doc.pages[0].text
