"""
Unit tests for document ingestion data models and identifier helpers.
"""
import pytest

from src.ingestion.models import (
    Chunk,
    Document,
    ExtractedPage,
    IngestionResult,
    IngestionStatus,
    compute_chunk_id,
    compute_sha256,
)


class TestExtractedPage:
    def test_valid_extracted_page_construction(self):
        page = ExtractedPage(page_number=1, text="This is page one text.")
        assert page.page_number == 1
        assert page.text == "This is page one text."
        assert page.has_text is True
        assert page.is_empty is False
        assert page.text_length == len("This is page one text.")
        assert page.char_count == len("This is page one text.")

    @pytest.mark.parametrize("invalid_page", [0, -1, -42])
    def test_reject_zero_and_negative_page_numbers(self, invalid_page):
        with pytest.raises(ValueError, match="page_number must be >= 1"):
            ExtractedPage(page_number=invalid_page, text="Sample text")

    def test_reject_non_int_page_number(self):
        with pytest.raises(TypeError, match="page_number must be an int"):
            ExtractedPage(page_number="1", text="Sample text")  # type: ignore
        with pytest.raises(TypeError, match="page_number must be an int"):
            ExtractedPage(page_number=True, text="Sample text")  # type: ignore

    def test_empty_and_whitespace_only_page_text(self):
        empty_page = ExtractedPage(page_number=1, text="")
        assert empty_page.has_text is False
        assert empty_page.is_empty is True
        assert empty_page.text_length == 0

        whitespace_page = ExtractedPage(page_number=2, text="   \n\t  ")
        assert whitespace_page.has_text is False
        assert whitespace_page.is_empty is True
        assert whitespace_page.text_length == 7

    def test_extracted_page_immutability(self):
        page = ExtractedPage(page_number=1, text="Immutable")
        with pytest.raises((AttributeError, TypeError)):
            page.text = "Changed"  # type: ignore


class TestDocumentFingerprint:
    def test_stable_fingerprint_for_identical_bytes(self):
        data = b"Simulated PDF content bytes for fingerprint test."
        hash1 = compute_sha256(data)
        hash2 = compute_sha256(data)
        assert hash1 == hash2
        assert len(hash1) == 64
        import hashlib
        assert hash1 == hashlib.sha256(data).hexdigest()

    def test_different_fingerprints_for_different_bytes(self):
        data_a = b"First document binary content."
        data_b = b"Second document binary content."
        assert compute_sha256(data_a) != compute_sha256(data_b)

    def test_fingerprint_type_validation(self):
        with pytest.raises(TypeError, match="content must be bytes"):
            compute_sha256("string not bytes")  # type: ignore


class TestDocument:
    def test_valid_document_metadata(self):
        p1 = ExtractedPage(page_number=1, text="Page 1")
        p2 = ExtractedPage(page_number=2, text="Page 2")
        doc = Document(
            doc_id="a1b2c3d4e5f6" * 5 + "1234",
            source_filename="test_manual.pdf",
            pages=[p1, p2],
        )
        assert doc.doc_id == "a1b2c3d4e5f6" * 5 + "1234"
        assert doc.source_filename == "test_manual.pdf"
        assert doc.file_name == "test_manual.pdf"
        assert doc.total_pages == 2
        assert len(doc.pages) == 2

    def test_document_enforces_ascending_page_ordering(self):
        p1 = ExtractedPage(page_number=1, text="Page 1")
        p2 = ExtractedPage(page_number=2, text="Page 2")

        # Out-of-order pages
        with pytest.raises(ValueError, match="Pages must be ordered by ascending page_number"):
            Document(doc_id="abc12345", source_filename="file.pdf", pages=[p2, p1])

        # Duplicate page numbers
        p1_dup = ExtractedPage(page_number=1, text="Duplicate Page 1")
        with pytest.raises(ValueError, match="Duplicate page_number 1"):
            Document(doc_id="abc12345", source_filename="file.pdf", pages=[p1, p1_dup])

    def test_reject_empty_document_identifiers(self):
        with pytest.raises(ValueError, match="doc_id cannot be empty"):
            Document(doc_id="", source_filename="file.pdf")
        with pytest.raises(ValueError, match="source_filename cannot be empty"):
            Document(doc_id="abc12345", source_filename="  ")


class TestChunk:
    def test_valid_chunk_construction(self):
        chunk_id = compute_chunk_id(doc_id="doc123", page_number=1, chunk_index=0)
        chunk = Chunk(
            chunk_id=chunk_id,
            doc_id="doc123",
            source_filename="guide.pdf",
            page_number=1,
            chunk_index=0,
            text="This is a coherent knowledge chunk.",
        )
        assert chunk.chunk_id == "doc123_p1_c0"
        assert chunk.doc_id == "doc123"
        assert chunk.source_filename == "guide.pdf"
        assert chunk.page_number == 1
        assert chunk.chunk_index == 0
        assert chunk.text == "This is a coherent knowledge chunk."
        assert chunk.char_count == len("This is a coherent knowledge chunk.")

    @pytest.mark.parametrize("empty_text", ["", "   ", "\n\t  \n"])
    def test_reject_empty_or_whitespace_chunk_text(self, empty_text):
        with pytest.raises(ValueError, match="chunk text must be a non-empty, non-whitespace string"):
            Chunk(
                chunk_id="c1",
                doc_id="doc1",
                source_filename="f.pdf",
                page_number=1,
                chunk_index=0,
                text=empty_text,
            )

    @pytest.mark.parametrize("invalid_page", [0, -1, -5])
    def test_reject_invalid_chunk_page_number(self, invalid_page):
        with pytest.raises(ValueError, match="page_number must be >= 1"):
            Chunk(
                chunk_id="c1",
                doc_id="doc1",
                source_filename="f.pdf",
                page_number=invalid_page,
                chunk_index=0,
                text="Valid chunk text",
            )

    @pytest.mark.parametrize("invalid_index", [-1, -99])
    def test_reject_invalid_chunk_index(self, invalid_index):
        with pytest.raises(ValueError, match="chunk_index must be >= 0"):
            Chunk(
                chunk_id="c1",
                doc_id="doc1",
                source_filename="f.pdf",
                page_number=1,
                chunk_index=invalid_index,
                text="Valid chunk text",
            )

    def test_reject_empty_chunk_metadata_fields(self):
        with pytest.raises(ValueError, match="doc_id must be a non-empty string"):
            Chunk(chunk_id="c1", doc_id="", source_filename="f.pdf", page_number=1, chunk_index=0, text="valid")
        with pytest.raises(ValueError, match="source_filename must be a non-empty string"):
            Chunk(chunk_id="c1", doc_id="doc1", source_filename=" ", page_number=1, chunk_index=0, text="valid")
        with pytest.raises(ValueError, match="chunk_id must be a non-empty string"):
            Chunk(chunk_id=" ", doc_id="doc1", source_filename="f.pdf", page_number=1, chunk_index=0, text="valid")


class TestChunkIdComputation:
    def test_deterministic_chunk_ids(self):
        id1 = compute_chunk_id(doc_id="hash789", page_number=3, chunk_index=4)
        id2 = compute_chunk_id(doc_id="hash789", page_number=3, chunk_index=4)
        assert id1 == id2 == "hash789_p3_c4"

    def test_chunk_id_varies_with_metadata(self):
        base = compute_chunk_id(doc_id="hash789", page_number=1, chunk_index=0)
        diff_page = compute_chunk_id(doc_id="hash789", page_number=2, chunk_index=0)
        diff_chunk = compute_chunk_id(doc_id="hash789", page_number=1, chunk_index=1)
        diff_doc = compute_chunk_id(doc_id="otherhash", page_number=1, chunk_index=0)

        assert base != diff_page
        assert base != diff_chunk
        assert base != diff_doc

    def test_chunk_id_invalid_inputs(self):
        with pytest.raises(ValueError, match="doc_id cannot be empty"):
            compute_chunk_id(doc_id="", page_number=1, chunk_index=0)
        with pytest.raises(ValueError, match="page_number must be an integer >= 1"):
            compute_chunk_id(doc_id="doc1", page_number=0, chunk_index=0)
        with pytest.raises(ValueError, match="chunk_index must be an integer >= 0"):
            compute_chunk_id(doc_id="doc1", page_number=1, chunk_index=-1)


class TestIngestionResult:
    def test_all_ingestion_statuses_defined(self):
        expected_statuses = {
            "success": IngestionStatus.SUCCESS,
            "partial_success": IngestionStatus.PARTIAL,
            "no_extractable_text": IngestionStatus.NO_TEXT,
            "empty_document": IngestionStatus.EMPTY,
            "encrypted_document": IngestionStatus.ENCRYPTED,
            "failed_corrupted": IngestionStatus.FAILED,
        }
        for val, enum_member in expected_statuses.items():
            assert enum_member.value == val

    def test_successful_ingestion_result(self):
        chunk = Chunk(
            chunk_id="d1_p1_c0",
            doc_id="d1",
            source_filename="doc.pdf",
            page_number=1,
            chunk_index=0,
            text="Knowledge chunk text",
        )
        page = ExtractedPage(page_number=1, text="Full page text")
        result = IngestionResult(
            status=IngestionStatus.SUCCESS,
            doc_id="d1",
            source_filename="doc.pdf",
            total_pages=1,
            pages=[page],
            chunks=[chunk],
        )
        assert result.status == IngestionStatus.SUCCESS
        assert result.is_usable is True
        assert len(result.extracted_chunks) == 1
        assert result.error_message is None
        assert len(result.warnings) == 0

    def test_partial_ingestion_is_still_usable(self):
        """A page without extractable text must not prevent an otherwise usable document from succeeding."""
        p1 = ExtractedPage(page_number=1, text="Text from page 1")
        p2 = ExtractedPage(page_number=2, text="")  # blank page
        chunk = Chunk(
            chunk_id="d1_p1_c0",
            doc_id="d1",
            source_filename="mixed.pdf",
            page_number=1,
            chunk_index=0,
            text="Text from page 1",
        )
        result = IngestionResult(
            status=IngestionStatus.PARTIAL,
            doc_id="d1",
            source_filename="mixed.pdf",
            total_pages=2,
            pages=[p1, p2],
            chunks=[chunk],
            warnings=["Page 2 contains no extractable text"],
        )
        assert result.status == IngestionStatus.PARTIAL
        assert result.is_usable is True
        assert len(result.warnings) == 1

    def test_unusable_ingestion_results(self):
        no_text_result = IngestionResult(
            status=IngestionStatus.NO_TEXT,
            doc_id="d2",
            source_filename="scanned.pdf",
            total_pages=5,
            warnings=["Document appears to be scanned with no OCR layer"],
        )
        assert no_text_result.is_usable is False

        encrypted_result = IngestionResult(
            status=IngestionStatus.ENCRYPTED,
            source_filename="protected.pdf",
            error_message="Document is password encrypted",
        )
        assert encrypted_result.is_usable is False

        failed_result = IngestionResult(
            status=IngestionStatus.FAILED,
            source_filename="corrupt.pdf",
            error_message="Corrupted PDF header",
        )
        assert failed_result.is_usable is False

    def test_invalid_status_type(self):
        with pytest.raises(TypeError, match="status must be an IngestionStatus"):
            IngestionResult(status="success")  # type: ignore
