"""Real end-to-end RAG integration test using real Ollama inference (nomic-embed-text and llama3.2:1b)."""
from __future__ import annotations

import asyncio
from dataclasses import replace
import json
from pathlib import Path
import shutil
import sys
import tempfile

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import fitz
import httpx

import src.api.main as api
from config import settings
from src.storage.vector_index import VectorIndex


def create_sample_pdf(pdf_path: Path):
    doc = fitz.open()
    
    # Page 1
    p1 = doc.new_page()
    p1.insert_text(fitz.Point(50, 72), "Project name: LocalRAG.\nProject purpose: Offline document intelligence.")
    
    # Page 2
    p2 = doc.new_page()
    p2.insert_text(fitz.Point(50, 72), "Project deadline: 25 October 2026.\nBackend framework: FastAPI.")
    
    # Page 3
    p3 = doc.new_page()
    p3.insert_text(fitz.Point(50, 72), "Embedding model: nomic-embed-text.\nDefault chat model: llama3.2:1b.")
    
    doc.save(str(pdf_path))
    doc.close()


async def run_live_rag_integration():
    print("=" * 70)
    print("STARTING REAL LIVE RAG INTEGRATION VERIFICATION")
    print("Ollama Base URL:", settings.ollama_base_url)
    print("Embedding Model:", settings.embedding_model)
    print("Chat Model:     ", settings.chat_model_default)
    print("Similarity Thr: ", settings.similarity_threshold)
    print("=" * 70)

    # 1. Setup isolated test directory
    test_dir = PROJECT_ROOT / "data" / "tmp_live_rag_test"
    if test_dir.exists():
        shutil.rmtree(test_dir, ignore_errors=True)
    
    docs_dir = test_dir / "documents"
    idx_dir = test_dir / "index"
    docs_dir.mkdir(parents=True, exist_ok=True)
    idx_dir.mkdir(parents=True, exist_ok=True)

    # Patch api.settings to use temporary directory
    original_settings = api.settings
    test_settings = replace(
        settings,
        documents_dir=docs_dir,
        index_dir=idx_dir,
        similarity_threshold=0.55,
    )
    api.settings = test_settings

    pdf_filename = "project_specification.pdf"
    sample_pdf_path = test_dir / pdf_filename
    create_sample_pdf(sample_pdf_path)
    pdf_bytes = sample_pdf_path.read_bytes()

    verification_log = {}

    try:
        transport = httpx.ASGITransport(app=api.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            
            # --- Check readiness ---
            ready_resp = await client.get("/ready")
            print(f"GET /ready: status={ready_resp.status_code}, body={ready_resp.json()}")
            assert ready_resp.status_code == 200, f"Ready check failed: {ready_resp.text}"
            assert ready_resp.json().get("ollama") is True

            # --- Step A: Ingestion ---
            print("\n--- Step A: Real PDF Ingestion ---")
            upload_resp = await client.post(
                "/upload",
                files={"file": (pdf_filename, pdf_bytes, "application/pdf")},
            )
            print("Upload status:", upload_resp.status_code)
            upload_data = upload_resp.json()
            print("Upload payload:", json.dumps(upload_data, indent=2))
            assert upload_resp.status_code == 200, f"Upload failed: {upload_resp.text}"
            doc_id = upload_data["document_id"]
            num_chunks = upload_data["chunks"]
            assert num_chunks == 3, f"Expected 3 chunks (1 per page), got {num_chunks}"
            assert (docs_dir / pdf_filename).exists(), "Uploaded PDF not saved in documents_dir"
            assert (idx_dir / "index.npz").exists(), "index.npz not created"
            assert (idx_dir / "index.json").exists(), "index.json not created"

            verification_log["ingestion"] = {
                "document_id": doc_id,
                "filename": pdf_filename,
                "extracted_pages": 3,
                "chunks": num_chunks,
            }

            # --- Step B & C: Retrieval & Generation ---
            print("\n--- Step B & C: Real Retrieval & Answer Generation ---")
            q1 = "When is the project deadline?"
            query_resp = await client.post("/query", json={"question": q1})
            print("Query status:", query_resp.status_code)
            query_data = query_resp.json()
            print("Query Response:\n", json.dumps(query_data, indent=2))
            assert query_resp.status_code == 200, f"Query failed: {query_resp.text}"
            
            answer = query_data["answer"]
            sources = query_data["sources"]
            assert len(sources) > 0, "No sources retrieved for deadline question!"
            top_source = sources[0]
            print(f"Top source: file={top_source['filename']}, page={top_source['page']}, score={top_source['score']:.4f}")
            assert top_source["filename"] == pdf_filename
            assert top_source["page"] == 2, f"Expected page 2 for deadline, got {top_source['page']}"
            assert "25 October 2026" in top_source["text"]
            assert "25 October 2026" in answer, f"LLM did not state the deadline date: {answer}"

            verification_log["supported_query"] = {
                "question": q1,
                "answer": answer,
                "top_source_filename": top_source["filename"],
                "top_source_page": top_source["page"],
                "top_source_score": top_source["score"],
            }

            # --- Step D: Unsupported questions ---
            print("\n--- Step D: Unsupported Question Handling ---")
            q2 = "Who is the project manager?"
            unsupported_resp = await client.post("/query", json={"question": q2})
            unsupported_data = unsupported_resp.json()
            print("Unsupported Query Response:\n", json.dumps(unsupported_data, indent=2))
            assert unsupported_resp.status_code == 200
            ans2 = unsupported_data["answer"].lower()
            # Must not invent a name; must state insufficient evidence / cannot answer
            refusal_or_unsupported = any(
                phrase in ans2
                for phrase in (
                    "enough evidence",
                    "insufficient",
                    "not contain",
                    "not specified",
                    "not mention",
                    "does not state",
                    "no information",
                    "cannot answer",
                    "not provided",
                    "no direct",
                )
            )
            print("Safely handled without hallucination:", refusal_or_unsupported)
            assert refusal_or_unsupported, f"LLM may have hallucinated or answered ungrounded: {ans2}"

            verification_log["unsupported_query"] = {
                "question": q2,
                "answer": unsupported_data["answer"],
                "sources_count": len(unsupported_data["sources"]),
                "safely_handled": refusal_or_unsupported,
            }

            # --- Step E: Persistence across reinitialization ---
            print("\n--- Step E: Persistence Across Reinitialization ---")
            # Reload vector index directly from disk to simulate application restart
            reloaded_index = VectorIndex.load(
                idx_dir,
                test_settings.embedding_model,
                test_settings.chunk_size,
                test_settings.chunk_overlap,
            )
            assert len(reloaded_index.chunks) == 3, f"Expected 3 chunks after reload, got {len(reloaded_index.chunks)}"
            assert reloaded_index.vectors.shape == (3, 768), f"Expected vectors shape (3, 768), got {reloaded_index.vectors.shape}"

            # Query through API again
            query_resp_restart = await client.post("/query", json={"question": q1})
            assert query_resp_restart.status_code == 200
            assert "25 October 2026" in query_resp_restart.json()["answer"]
            print("Persistence verification: PASSED (retrieval and generation work post-restart)")

            verification_log["persistence"] = {
                "reloaded_chunks": len(reloaded_index.chunks),
                "reloaded_vectors_shape": list(reloaded_index.vectors.shape),
                "answer_after_restart": query_resp_restart.json()["answer"],
            }

            # --- Step F: Document deletion ---
            print("\n--- Step F: Document Deletion ---")
            del_resp = await client.delete(f"/documents/{doc_id}")
            print(f"DELETE /documents/{doc_id}: status={del_resp.status_code}, body={del_resp.json()}")
            assert del_resp.status_code == 200
            assert del_resp.json()["removed_chunks"] == 3
            assert not (docs_dir / pdf_filename).exists(), "PDF file still exists after deletion"

            # Check index on disk
            empty_index = VectorIndex.load(
                idx_dir,
                test_settings.embedding_model,
                test_settings.chunk_size,
                test_settings.chunk_overlap,
            )
            assert len(empty_index.chunks) == 0, f"Expected 0 chunks after delete, got {len(empty_index.chunks)}"
            assert empty_index.vectors.shape[0] == 0, "Vectors still present after delete"

            # Query should now return no evidence
            post_del_query = await client.post("/query", json={"question": q1})
            assert post_del_query.status_code == 200
            assert post_del_query.json()["sources"] == [], "Sources returned for deleted document"
            print("Deletion verification: PASSED (chunks, vectors, and files completely purged)")

            verification_log["deletion"] = {
                "removed_chunks": del_resp.json()["removed_chunks"],
                "index_chunks_after_delete": len(empty_index.chunks),
                "sources_after_delete": len(post_del_query.json()["sources"]),
            }

    finally:
        # Restore api.settings
        api.settings = original_settings
        # Clean up temporary test directory
        shutil.rmtree(test_dir, ignore_errors=True)
        print(f"\nTemporary test directory {test_dir} cleaned up successfully.")

    # Save verification log to data/live_rag_verification.json
    out_file = PROJECT_ROOT / "data" / "live_rag_verification.json"
    out_file.write_text(json.dumps(verification_log, indent=2), encoding="utf-8")
    print(f"Live verification results saved to {out_file}")
    print("\nALL TASK 5 INTEGRATION CHECKS PASSED WITH REAL OLLAMA INFERENCE!")


if __name__ == "__main__":
    asyncio.run(run_live_rag_integration())
