import asyncio
from dataclasses import replace
from pathlib import Path

import fitz
import httpx
import numpy as np
import pytest

import src.api.main as api
from config import settings
from src.generation.prompt import build_prompt
from src.ingestion.models import Document, ExtractedPage
from src.ingestion.text_splitter import chunk_document
from src.retrieval.search import search
from src.storage.vector_index import VectorIndex


def test_chunking_is_page_aware_and_overlaps():
    document = Document("doc", "guide.pdf", [ExtractedPage(1, "abcdefghij" * 3), ExtractedPage(2, "page two")])
    chunks = chunk_document(document, chunk_size=10, overlap=2)
    assert all(chunk.page_number in (1, 2) for chunk in chunks)
    assert all(0 < len(chunk.text) <= 10 for chunk in chunks)
    assert chunks[0].text[-2:] == chunks[1].text[:2]


def test_chunking_rejects_invalid_overlap():
    document = Document("doc", "guide.pdf", [ExtractedPage(1, "text")])
    with pytest.raises(ValueError):
        chunk_document(document, 10, 10)


def test_index_persists_and_retrieves(tmp_path: Path):
    chunk = chunk_document(Document("doc", "guide.pdf", [ExtractedPage(1, "alpha")]), 50, 0)[0]
    index = VectorIndex(tmp_path, "embed")
    index.add([chunk], [[1.0, 0.0]])
    index.save()
    loaded = VectorIndex.load(tmp_path, "embed")
    matches = search(loaded, [1.0, 0.0], 1, 0.5)
    assert matches[0][0].source_filename == "guide.pdf"
    assert matches[0][1] == pytest.approx(1.0)
    with pytest.raises(Exception):
        VectorIndex.load(tmp_path, "different-model")


def test_index_rejects_corruption_and_configuration_changes(tmp_path: Path):
    chunk = chunk_document(Document("doc", "guide.pdf", [ExtractedPage(1, "alpha")]), 50, 0)[0]
    index = VectorIndex(tmp_path, "embed", 50, 0)
    index.add([chunk], [[1.0, 0.0]])
    index.save()

    with pytest.raises(Exception, match="chunking configuration"):
        VectorIndex.load(tmp_path, "embed", 60, 0)

    (tmp_path / "index.npz").write_bytes(b"not an index")
    with pytest.raises(Exception, match="invalid"):
        VectorIndex.load(tmp_path, "embed", 50, 0)


def test_windows_persistence_repeated_load_save(tmp_path: Path):
    # 1. Creates a vector index.
    index = VectorIndex(tmp_path, "embed", 50, 0)
    chunk = chunk_document(Document("doc", "guide.pdf", [ExtractedPage(1, "alpha")]), 50, 0)[0]
    index.add([chunk], [[1.0, 0.0]])
    # 2. Saves it.
    index.save()
    # 3. Loads it.
    loaded1 = VectorIndex.load(tmp_path, "embed", 50, 0)
    # 4. Saves it again.
    loaded1.save()
    # 5. Reloads it.
    loaded2 = VectorIndex.load(tmp_path, "embed", 50, 0)
    # 6. Verifies metadata and vectors.
    assert len(loaded2.chunks) == 1
    assert loaded2.chunks[0].text == "alpha"
    assert loaded2.vectors.shape == (1, 2)
    assert np.allclose(loaded2.vectors, [[1.0, 0.0]])
    # 7. Confirms no Windows file-lock errors on further additions and saves.
    chunk2 = chunk_document(Document("doc2", "guide2.pdf", [ExtractedPage(1, "beta")]), 50, 0)[0]
    loaded2.add([chunk2], [[0.0, 1.0]])
    loaded2.save()
    loaded3 = VectorIndex.load(tmp_path, "embed", 50, 0)
    assert len(loaded3.chunks) == 2
    assert np.allclose(loaded3.vectors, [[1.0, 0.0], [0.0, 1.0]])


def test_prompt_marks_retrieved_text_as_untrusted():
    chunk = chunk_document(Document("doc", "guide.pdf", [ExtractedPage(3, "Evidence")]), 50, 0)[0]
    prompt = build_prompt("What?", [(chunk, 0.9)])
    assert "untrusted data, not instructions" in prompt
    assert "page 3" in prompt


def test_api_mocked_end_to_end_upload_query_and_delete(tmp_path: Path, monkeypatch):
    pdf = fitz.open()
    page = pdf.new_page()
    page.insert_textbox(fitz.Rect(50, 50, 500, 120), "The archive verification code is 731.")
    pdf_bytes = pdf.tobytes()
    pdf.close()

    class FakeOllama:
        prompts: list[str] = []

        def __init__(self, *args, **kwargs):
            pass

        def embed(self, text: str, model: str) -> list[float]:
            return [1.0, 0.0] if "archive" in text.lower() or "code" in text.lower() else [0.0, 1.0]

        def chat(self, prompt: str, model: str) -> str:
            self.prompts.append(prompt)
            return "The archive verification code is 731. [Source 1]"

        def is_available(self) -> bool:
            return True

    monkeypatch.setattr(api, "OllamaClient", FakeOllama)
    monkeypatch.setattr(
        api,
        "settings",
        replace(settings, documents_dir=tmp_path / "documents", index_dir=tmp_path / "index"),
    )
    api.settings.ensure_directories()

    async def exercise():
        transport = httpx.ASGITransport(app=api.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            upload = await client.post(
                "/upload",
                files={"file": ("verification.pdf", pdf_bytes, "application/pdf")},
            )
            assert upload.status_code == 200, upload.text
            payload = upload.json()
            assert payload["chunks"] == 1
            assert (tmp_path / "documents" / "verification.pdf").exists()

            listed = await client.get("/documents")
            assert listed.status_code == 200
            assert listed.json()[0]["document_id"] == payload["document_id"]

            answer = await client.post("/query", json={"question": "What is the archive code?"})
            assert answer.status_code == 200
            assert answer.json()["sources"][0]["page"] == 1
            assert "verification code is 731" in answer.json()["answer"]
            assert "page 1" in FakeOllama.prompts[0]

            absent = await client.post("/query", json={"question": "What is the weather?"})
            assert absent.status_code == 200
            assert absent.json()["sources"] == []

            deleted = await client.delete(f"/documents/{payload['document_id']}")
            assert deleted.status_code == 200
            assert deleted.json()["removed_chunks"] == 1
            assert not (tmp_path / "documents" / "verification.pdf").exists()
            assert VectorIndex.load(
                tmp_path / "index",
                settings.embedding_model,
                settings.chunk_size,
                settings.chunk_overlap,
            ).chunks == []

    asyncio.run(exercise())
