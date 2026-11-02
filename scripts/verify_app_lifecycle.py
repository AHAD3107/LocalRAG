"""Application lifecycle verification: startup, ingestion, query, restart, re-query, deletion."""
from __future__ import annotations

import asyncio
from dataclasses import replace
import json
from pathlib import Path
import shutil
import sys

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import fitz
import httpx

import src.api.main as api
from config import settings
from src.storage.vector_index import VectorIndex


def create_lifecycle_pdf(pdf_path: Path):
    doc = fitz.open()
    p1 = doc.new_page()
    p1.insert_text(fitz.Point(50, 72), "Project name: LocalRAG.\nProject purpose: Private document search.")
    p2 = doc.new_page()
    p2.insert_text(fitz.Point(50, 72), "Target release date: 15 November 2026.\nRuntime host: Windows 11.")
    doc.save(str(pdf_path))
    doc.close()


async def run_lifecycle_verification():
    print("=" * 70)
    print("STARTING FULL APPLICATION LIFECYCLE & RESTART VERIFICATION")
    print("=" * 70)

    test_dir = PROJECT_ROOT / "data" / "tmp_lifecycle_test"
    if test_dir.exists():
        shutil.rmtree(test_dir, ignore_errors=True)

    docs_dir = test_dir / "documents"
    idx_dir = test_dir / "index"
    docs_dir.mkdir(parents=True, exist_ok=True)
    idx_dir.mkdir(parents=True, exist_ok=True)

    original_settings = api.settings
    test_settings = replace(
        settings,
        documents_dir=docs_dir,
        index_dir=idx_dir,
        similarity_threshold=0.55,
    )
    api.settings = test_settings

    pdf_filename = "lifecycle_spec.pdf"
    sample_pdf_path = test_dir / pdf_filename
    create_lifecycle_pdf(sample_pdf_path)
    pdf_bytes = sample_pdf_path.read_bytes()

    lifecycle_log = {}

    try:
        # Phase 1: Initial Application Run
        print("\n[Phase 1] Starting Initial Backend Session...")
        transport = httpx.ASGITransport(app=api.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            # 1. Health check
            health = await client.get("/health")
            assert health.status_code == 200
            print("1. Backend Health Check: OK")

            # 2. Upload real PDF
            upload_resp = await client.post(
                "/upload",
                files={"file": (pdf_filename, pdf_bytes, "application/pdf")},
            )
            assert upload_resp.status_code == 200, f"Upload failed: {upload_resp.text}"
            upload_data = upload_resp.json()
            doc_id = upload_data["document_id"]
            chunks = upload_data["chunks"]
            print(f"2. PDF Uploaded: doc_id={doc_id}, chunks={chunks}")
            assert chunks == 2

            # 3. List documents (as frontend dropdown does)
            docs_list = await client.get("/documents")
            assert docs_list.status_code == 200
            doc_items = docs_list.json()
            print(f"3. Listed Documents: {doc_items}")
            assert len(doc_items) == 1
            assert doc_items[0]["filename"] == pdf_filename

            # 4. Ask question & verify source citation
            q = "What is the target release date?"
            query_resp = await client.post("/query", json={"question": q, "model": "llama3.2:1b"})
            assert query_resp.status_code == 200
            query_data = query_resp.json()
            print(f"4. Query Answer: {query_data['answer']}")
            assert "15 November 2026" in query_data["answer"]
            assert len(query_data["sources"]) > 0
            src = query_data["sources"][0]
            print(f"   Source Citation: file={src['filename']}, page={src['page']}, score={src['score']:.4f}")
            assert src["filename"] == pdf_filename
            assert src["page"] == 2
            assert src["score"] >= 0.55

            lifecycle_log["initial_session"] = {
                "uploaded_doc_id": doc_id,
                "chunks": chunks,
                "question": q,
                "answer": query_data["answer"],
                "source_file": src["filename"],
                "source_page": src["page"],
                "source_score": src["score"],
            }

        # Phase 2: Application Restart Simulation
        print("\n[Phase 2] Simulating Application Shutdown and Restart...")
        # Simulating fresh restart: verify disk persistence directly
        restarted_index = VectorIndex.load(
            idx_dir,
            test_settings.embedding_model,
            test_settings.chunk_size,
            test_settings.chunk_overlap,
        )
        assert len(restarted_index.chunks) == 2
        print(f"5. Reloaded index from disk: {len(restarted_index.chunks)} chunks, shape={restarted_index.vectors.shape}")

        # Start a brand new client session against the app
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=api.app), base_url="http://test") as restart_client:
            # 6. Confirm document is still listed post-restart
            restarted_docs = await restart_client.get("/documents")
            assert restarted_docs.status_code == 200
            restarted_items = restarted_docs.json()
            print(f"6. Post-restart Document List: {restarted_items}")
            assert len(restarted_items) == 1
            assert restarted_items[0]["document_id"] == doc_id

            # 7. Ask same question again & confirm correct retrieval
            re_query = await restart_client.post("/query", json={"question": q, "model": "llama3.2:1b"})
            assert re_query.status_code == 200
            re_data = re_query.json()
            print(f"7. Post-restart Query Answer: {re_data['answer']}")
            assert "15 November 2026" in re_data["answer"]
            assert re_data["sources"][0]["page"] == 2

            lifecycle_log["restarted_session"] = {
                "listed_docs_count": len(restarted_items),
                "re_query_answer": re_data["answer"],
                "source_page": re_data["sources"][0]["page"],
            }

        # Phase 3: Document Deletion
        print("\n[Phase 3] Document Deletion Verification...")
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=api.app), base_url="http://test") as delete_client:
            del_resp = await delete_client.delete(f"/documents/{doc_id}")
            assert del_resp.status_code == 200
            print(f"8. Deleted document: removed_chunks={del_resp.json()['removed_chunks']}")
            assert del_resp.json()["removed_chunks"] == 2

            # Confirm file unlinked
            assert not (docs_dir / pdf_filename).exists()
            print("9. Confirmed PDF file unlinked from documents directory.")

            # Confirm documents list is empty
            post_del_docs = await delete_client.get("/documents")
            assert post_del_docs.json() == []
            print("10. Confirmed document list is now empty.")

            # Confirm query returns no evidence
            post_del_query = await delete_client.post("/query", json={"question": q})
            assert post_del_query.status_code == 200
            assert post_del_query.json()["sources"] == []
            print("11. Confirmed query returns 0 sources post-deletion.")

            lifecycle_log["deletion"] = {
                "removed_chunks": del_resp.json()["removed_chunks"],
                "post_deletion_doc_count": len(post_del_docs.json()),
                "post_deletion_sources": len(post_del_query.json()["sources"]),
            }

    finally:
        api.settings = original_settings
        shutil.rmtree(test_dir, ignore_errors=True)
        print(f"\nCleaned up temporary directory: {test_dir}")

    out_path = PROJECT_ROOT / "data" / "app_lifecycle_verification.json"
    out_path.write_text(json.dumps(lifecycle_log, indent=2), encoding="utf-8")
    print(f"Lifecycle verification log saved to: {out_path}")
    print("\nALL LIFECYCLE AND RESTART VERIFICATION CHECKS PASSED!")


if __name__ == "__main__":
    asyncio.run(run_lifecycle_verification())

