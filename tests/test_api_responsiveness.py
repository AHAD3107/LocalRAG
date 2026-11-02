"""Tests verifying API responsiveness during multi-chunk indexing."""
import asyncio
from dataclasses import replace
from pathlib import Path
import time

import fitz
import httpx
import pytest

import src.api.main as api
from config import settings


def test_api_remains_responsive_during_multi_chunk_upload(tmp_path: Path, monkeypatch):
    """Verify that long-running chunk embedding operations do not block the event loop."""
    # Create a 3-page test PDF
    pdf = fitz.open()
    for i in range(1, 4):
        page = pdf.new_page()
        page.insert_textbox(fitz.Rect(50, 50, 400, 200), f"Section {i}: Content for page {i}.")
    pdf_bytes = pdf.tobytes()
    pdf.close()

    class SimulatingSlowOllama:
        def __init__(self, *args, **kwargs):
            pass

        def embed(self, text: str, model: str) -> list[float]:
            # Simulate a 100ms processing delay per chunk
            time.sleep(0.1)
            return [0.1] * 768

        def is_available(self, *args, **kwargs) -> bool:
            return True

    monkeypatch.setattr(api, "OllamaClient", SimulatingSlowOllama)
    monkeypatch.setattr(
        api,
        "settings",
        replace(settings, documents_dir=tmp_path / "documents", index_dir=tmp_path / "index"),
    )
    api.settings.ensure_directories()

    async def exercise():
        transport = httpx.ASGITransport(app=api.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            # Start upload task (3 pages = 3 chunks = ~300ms total)
            upload_task = asyncio.create_task(
                client.post("/upload", files={"file": ("slow_doc.pdf", pdf_bytes, "application/pdf")})
            )

            # Wait briefly to ensure upload has started its embedding loop
            await asyncio.sleep(0.05)

            # Send a concurrent health check while upload is still in progress
            t0 = time.perf_counter()
            health_resp = await client.get("/health")
            elapsed = time.perf_counter() - t0

            # Health check must respond immediately without being blocked
            assert health_resp.status_code == 200
            assert health_resp.json() == {"status": "ok"}
            assert elapsed < 0.08, f"Health check was blocked by upload! Took {elapsed:.3f}s"

            # Wait for upload to complete successfully
            upload_resp = await upload_task
            assert upload_resp.status_code == 200
            assert upload_resp.json()["chunks"] == 3

    asyncio.run(exercise())

