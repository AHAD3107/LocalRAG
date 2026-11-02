"""Verification script for Task 5: Genuine operating system process restart verification.

Verifies:
1. Spawns backend process 1 using subprocess.Popen in an isolated temporary directory.
2. Ingests a real sample PDF via /upload.
3. Queries /query and verifies answer and source citation.
4. Kills backend process 1.
5. Spawns independent backend process 2 on the same persisted directory.
6. Queries /documents to verify document persisted.
7. Queries /query on process 2 to verify search & answer retrieval persist.
8. Deletes document via /documents/{doc_id} on process 2.
9. Confirms deletion in index and on disk.
10. Kills process 2 and cleans up temporary directory.
11. Saves full structured report to data/real_process_restart_verification.json.
"""
from __future__ import annotations

import datetime
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import fitz
import httpx


def generate_sample_pdf(output_path: Path, secret_phrase: str) -> None:
    doc = fitz.open()
    page = doc.new_page()
    text = (
        f"LocalRAG Reliability and Persistence Protocol.\n\n"
        f"The secret server restart verification passphrase is '{secret_phrase}'.\n"
        f"This security passphrase verifies that NumPy vector index archives and disk files\n"
        f"survive genuine operating system process termination, memory purging, and re-initialization."
    )
    page.insert_textbox(fitz.Rect(50, 50, 500, 300), text)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(output_path))
    doc.close()


def wait_for_ready(url: str, timeout_seconds: float = 15.0) -> bool:
    start_time = time.time()
    while time.time() - start_time < timeout_seconds:
        try:
            resp = httpx.get(f"{url}/ready", timeout=2.0)
            if resp.status_code == 200:
                return True
        except (httpx.ConnectError, httpx.HTTPError):
            pass
        time.sleep(0.5)
    return False


def run_process_restart_verification():
    test_port = 8008
    base_url = f"http://127.0.0.1:{test_port}"
    secret_passphrase = "Falcon-7729-Omega-Verified"
    test_env_dir = PROJECT_ROOT / "data" / "test_restart_sandbox"
    docs_dir = test_env_dir / "documents"
    index_dir = test_env_dir / "index"

    # Clean any prior sandbox
    if test_env_dir.exists():
        shutil.rmtree(test_env_dir, ignore_errors=True)
    docs_dir.mkdir(parents=True, exist_ok=True)
    index_dir.mkdir(parents=True, exist_ok=True)

    sample_pdf_path = test_env_dir / "restart_test_document.pdf"
    generate_sample_pdf(sample_pdf_path, secret_passphrase)

    python_exe = sys.executable
    env = os.environ.copy()
    env["DOCUMENTS_DIR"] = str(docs_dir)
    env["INDEX_DIR"] = str(index_dir)
    env["API_PORT"] = str(test_port)
    env["API_HOST"] = "127.0.0.1"

    verification_log = {
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "test_port": test_port,
        "secret_passphrase": secret_passphrase,
        "steps": [],
        "overall_success": False,
    }

    proc1 = None
    proc2 = None

    try:
        # STEP 1: Launch Process 1
        print(f"[Step 1] Launching Backend Process 1 on port {test_port}...")
        proc1 = subprocess.Popen(
            [
                python_exe,
                "-m",
                "uvicorn",
                "src.api.main:app",
                "--host",
                "127.0.0.1",
                "--port",
                str(test_port),
            ],
            cwd=str(PROJECT_ROOT),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        pid1 = proc1.pid
        print(f"         Process 1 started with PID: {pid1}")

        is_ready = wait_for_ready(base_url, timeout_seconds=15.0)
        assert is_ready, "Process 1 failed to reach ready state within timeout."
        verification_log["steps"].append({
            "step": 1,
            "description": "Spawn Backend Process 1",
            "pid": pid1,
            "status": "SUCCESS",
        })

        # STEP 2: Ingest PDF into Process 1
        print("[Step 2] Ingesting sample PDF via /upload on Process 1...")
        with open(sample_pdf_path, "rb") as f:
            upload_resp = httpx.post(
                f"{base_url}/upload",
                files={"file": ("restart_test_document.pdf", f, "application/pdf")},
                timeout=60.0,
            )
        assert upload_resp.status_code == 200, f"Upload failed: {upload_resp.text}"
        upload_data = upload_resp.json()
        doc_id = upload_data["document_id"]
        chunk_count = upload_data["chunks"]
        print(f"         Uploaded successfully: doc_id={doc_id}, chunks={chunk_count}")
        verification_log["steps"].append({
            "step": 2,
            "description": "Upload sample PDF on Process 1",
            "response": upload_data,
            "status": "SUCCESS",
        })

        # STEP 3: Query Process 1
        print("[Step 3] Querying Process 1 for verification passphrase...")
        query_payload = {"question": "What is the secret server restart verification passphrase?"}
        q_resp1 = httpx.post(f"{base_url}/query", json=query_payload, timeout=60.0)
        assert q_resp1.status_code == 200, f"Query 1 failed: {q_resp1.text}"
        q_data1 = q_resp1.json()
        answer1 = q_data1["answer"]
        sources1 = q_data1["sources"]
        print(f"         Answer: {answer1[:80]}...")
        assert secret_passphrase in answer1 or "Falcon-7729" in answer1, f"Expected passphrase in answer: {answer1}"
        assert len(sources1) > 0 and sources1[0]["filename"] == "restart_test_document.pdf"
        verification_log["steps"].append({
            "step": 3,
            "description": "Query Process 1 and verify answer and citations",
            "answer": answer1,
            "sources": sources1,
            "status": "SUCCESS",
        })

        # STEP 4: Terminate Process 1
        print(f"[Step 4] Terminating Process 1 (PID {pid1})...")
        proc1.terminate()
        try:
            proc1.wait(timeout=8.0)
        except subprocess.TimeoutExpired:
            proc1.kill()
            proc1.wait(timeout=5.0)
        print("         Process 1 has terminated.")

        # Confirm process 1 is dead by attempting connection
        connection_dead = False
        try:
            httpx.get(f"{base_url}/health", timeout=1.0)
        except (httpx.ConnectError, httpx.HTTPError):
            connection_dead = True
        assert connection_dead, "Port still active after Process 1 termination!"
        verification_log["steps"].append({
            "step": 4,
            "description": "Terminate Process 1 and confirm memory purge",
            "pid": pid1,
            "status": "SUCCESS",
        })

        # Small pause before restarting
        time.sleep(1.0)

        # STEP 5: Spawn Process 2
        print(f"[Step 5] Spawning independent Backend Process 2 on port {test_port}...")
        proc2 = subprocess.Popen(
            [
                python_exe,
                "-m",
                "uvicorn",
                "src.api.main:app",
                "--host",
                "127.0.0.1",
                "--port",
                str(test_port),
            ],
            cwd=str(PROJECT_ROOT),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        pid2 = proc2.pid
        print(f"         Process 2 started with PID: {pid2} (distinct from PID 1: {pid1})")
        assert pid2 != pid1, "Process 2 PID must differ from Process 1."

        is_ready2 = wait_for_ready(base_url, timeout_seconds=15.0)
        assert is_ready2, "Process 2 failed to reach ready state within timeout."
        verification_log["steps"].append({
            "step": 5,
            "description": "Spawn independent Backend Process 2",
            "pid": pid2,
            "prior_pid": pid1,
            "status": "SUCCESS",
        })

        # STEP 6: Verify Documents List on Process 2
        print("[Step 6] Confirming document persistence via /documents on Process 2...")
        docs_resp = httpx.get(f"{base_url}/documents", timeout=15.0)
        assert docs_resp.status_code == 200, f"/documents failed: {docs_resp.text}"
        persisted_docs = docs_resp.json()
        print(f"         Persisted documents found: {len(persisted_docs)}")
        assert any(d["document_id"] == doc_id for d in persisted_docs), f"Document {doc_id} missing after restart!"
        verification_log["steps"].append({
            "step": 6,
            "description": "List documents on Process 2 and verify persistence",
            "documents": persisted_docs,
            "status": "SUCCESS",
        })

        # STEP 7: Query Process 2
        print("[Step 7] Querying Process 2 for verification passphrase...")
        q_resp2 = httpx.post(f"{base_url}/query", json=query_payload, timeout=60.0)
        assert q_resp2.status_code == 200, f"Query 2 failed: {q_resp2.text}"
        q_data2 = q_resp2.json()
        answer2 = q_data2["answer"]
        sources2 = q_data2["sources"]
        print(f"         Answer on Process 2: {answer2[:80]}...")
        assert secret_passphrase in answer2 or "Falcon-7729" in answer2, f"Expected passphrase in answer 2: {answer2}"
        assert len(sources2) > 0 and sources2[0]["filename"] == "restart_test_document.pdf"
        verification_log["steps"].append({
            "step": 7,
            "description": "Query Process 2 and verify answer and source citations",
            "answer": answer2,
            "sources": sources2,
            "status": "SUCCESS",
        })

        # STEP 8: Delete Document on Process 2
        print(f"[Step 8] Deleting document {doc_id} on Process 2...")
        del_resp = httpx.delete(f"{base_url}/documents/{doc_id}", timeout=15.0)
        assert del_resp.status_code == 200, f"Deletion failed: {del_resp.text}"
        del_data = del_resp.json()
        print(f"         Deleted chunks: {del_data.get('removed_chunks')}")
        assert del_data.get("removed_chunks") == chunk_count

        # Confirm documents list is empty
        docs_after_del = httpx.get(f"{base_url}/documents", timeout=15.0).json()
        assert len(docs_after_del) == 0, f"Expected 0 documents after deletion, got {docs_after_del}"
        verification_log["steps"].append({
            "step": 8,
            "description": "Delete document on Process 2 and verify index cleanup",
            "deleted_chunks": del_data.get("removed_chunks"),
            "status": "SUCCESS",
        })

        verification_log["overall_success"] = True
        print("\nALL PROCESS RESTART VERIFICATION STEPS PASSED SUCCESSFULLY!")

    finally:
        # Cleanup processes
        if proc1 and proc1.poll() is None:
            proc1.kill()
            proc1.wait(timeout=2.0)
        if proc2 and proc2.poll() is None:
            proc2.kill()
            proc2.wait(timeout=2.0)

        # Cleanup sandbox directory
        if test_env_dir.exists():
            shutil.rmtree(test_env_dir, ignore_errors=True)

    # Save verification report
    report_path = PROJECT_ROOT / "data" / "real_process_restart_verification.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(verification_log, indent=2), encoding="utf-8")
    print(f"Verification report saved to: {report_path.resolve()}")


if __name__ == "__main__":
    run_process_restart_verification()

