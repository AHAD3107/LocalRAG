import os
import httpx
import pytest


def _response(payload, status_code=200):
    request = httpx.Request("GET", "http://test/documents")
    return httpx.Response(status_code, json=payload, request=request)


def test_get_api_url_local_defaults(monkeypatch):
    import frontend.app as frontend

    monkeypatch.delenv("API_URL", raising=False)
    monkeypatch.delenv("API_HOST", raising=False)
    monkeypatch.delenv("API_PORT", raising=False)

    assert frontend.get_api_url() == "http://127.0.0.1:8000"


def test_get_api_url_port_override(monkeypatch):
    import frontend.app as frontend

    monkeypatch.delenv("API_URL", raising=False)
    monkeypatch.setenv("API_PORT", "9000")
    monkeypatch.setenv("API_HOST", "127.0.0.1")

    assert frontend.get_api_url() == "http://127.0.0.1:9000"


def test_get_api_url_docker_precedence(monkeypatch):
    import frontend.app as frontend

    monkeypatch.setenv("API_URL", "http://api:8000")
    monkeypatch.setenv("API_PORT", "9000")

    assert frontend.get_api_url() == "http://api:8000"


def test_refresh_returns_choices_and_clears_scalar_selection(monkeypatch):
    import frontend.app as frontend

    monkeypatch.setattr(
        frontend.httpx,
        "get",
        lambda *args, **kwargs: _response(
            {
                "value": [
                    {
                        "filename": "machine_learning_algorithms.pdf",
                        "document_id": "doc-123",
                        "chunks": 35,
                    }
                ],
                "Count": 1,
            }
        ),
    )

    status, update = frontend.refresh_documents()

    assert status == "Loaded 1 indexed document(s)."
    assert update["choices"] == [("machine_learning_algorithms.pdf", "doc-123")]
    assert update["value"] == "doc-123"
    assert update["type"] == "value"
    assert isinstance(update["choices"][0][1], str)


def test_refresh_accepts_plain_list_response_for_backend_compatibility(monkeypatch):
    import frontend.app as frontend

    monkeypatch.setattr(
        frontend.httpx,
        "get",
        lambda *args, **kwargs: _response(
            [{"filename": "guide.pdf", "document_id": "doc-456", "chunks": 2}]
        ),
    )

    status, update = frontend.refresh_documents()

    assert status == "Loaded 1 indexed document(s)."
    assert update["choices"] == [("guide.pdf", "doc-456")]
    assert update["value"] == "doc-456"


def test_refresh_clears_dropdown_when_no_documents(monkeypatch):
    import frontend.app as frontend

    monkeypatch.setattr(frontend.httpx, "get", lambda *args, **kwargs: _response([]))

    status, update = frontend.refresh_documents()

    assert status == "No indexed documents."
    assert update["choices"] == []
    assert update["value"] is None
    assert update["type"] == "value"


def test_delete_passes_scalar_id_and_refreshes_choices(monkeypatch):
    import frontend.app as frontend
    deleted_ids = []

    def fake_delete(url, **kwargs):
        deleted_ids.append(url.rsplit("/", 1)[-1])
        return _response({}, status_code=200)

    monkeypatch.setattr(frontend.httpx, "delete", fake_delete)
    monkeypatch.setattr(
        frontend.httpx,
        "get",
        lambda *args, **kwargs: _response([]),
    )

    status, update = frontend.delete_document("doc-123", True)

    assert deleted_ids == ["doc-123"]
    assert status == "Document deleted. No indexed documents."
    assert update["choices"] == []
    assert update["value"] is None
    assert update["type"] == "value"


def test_refresh_reports_api_errors_and_clears_stale_selection(monkeypatch):
    import frontend.app as frontend

    def failed_get(*args, **kwargs):
        request = httpx.Request("GET", "http://test/documents")
        raise httpx.HTTPStatusError(
            "server error",
            request=request,
            response=httpx.Response(503, request=request),
        )

    monkeypatch.setattr(frontend.httpx, "get", failed_get)

    status, update = frontend.refresh_documents()

    assert status == "Could not load documents (HTTP 503)."
    assert update["choices"] == []
    assert update["value"] is None
    assert update["type"] == "value"


def test_duplicate_upload_explains_existing_document(monkeypatch):
    import frontend.app as frontend

    class Upload:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    request = httpx.Request("POST", "http://test/upload")
    response = httpx.Response(409, json={"detail": "A document with that filename already exists."}, request=request)

    monkeypatch.setattr(frontend, "refresh_documents", lambda: (
        "Loaded 1 indexed document(s).",
        frontend._dropdown_update([("existing.pdf", "doc-123")]),
    ))
    monkeypatch.setattr("builtins.open", lambda *args, **kwargs: Upload())
    monkeypatch.setattr(
        frontend.httpx,
        "post",
        lambda *args, **kwargs: (_ for _ in ()).throw(httpx.HTTPStatusError("conflict", request=request, response=response)),
    )

    status, update = frontend.upload_pdf("existing.pdf")

    assert "already indexed" in status
    assert "select it from the indexed collection" in status.lower()
    assert update["choices"] == [("existing.pdf", "doc-123")]
    assert update["value"] == "doc-123"


def test_chat_empty_message():
    import frontend.app as frontend

    msg, history, status = frontend.chat("   ", [["prior q", "prior a"]])
    assert msg == ""
    assert history == [["prior q", "prior a"]]
    assert "Please enter a valid question" in status


def test_chat_success(monkeypatch):
    import frontend.app as frontend

    posted_payload = {}

    def fake_post(url, **kwargs):
        nonlocal posted_payload
        posted_payload = kwargs.get("json", {})
        req = httpx.Request("POST", url)
        return httpx.Response(
            200,
            json={
                "answer": "The project deadline is 25 October 2026.",
                "sources": [
                    {"filename": "specs.pdf", "page": 2, "score": 0.705, "text": "Deadline: 25 October 2026."}
                ],
            },
            request=req,
        )

    monkeypatch.setattr(frontend.httpx, "post", fake_post)

    msg, history, sources = frontend.chat("When is deadline?", [], model="llama3.2:1b")
    assert msg == ""
    assert len(history) == 1
    assert history[0][0] == "When is deadline?"
    assert "25 October 2026" in history[0][1]
    assert posted_payload["model"] == "llama3.2:1b"
    assert "specs.pdf" in sources
    assert "Page 2" in sources


def test_chat_api_error_handling(monkeypatch):
    import frontend.app as frontend

    def fake_post(url, **kwargs):
        req = httpx.Request("POST", url)
        raise httpx.HTTPStatusError("Service unavailable", request=req, response=httpx.Response(503, request=req))

    monkeypatch.setattr(frontend.httpx, "post", fake_post)

    msg, history, sources = frontend.chat("Any question?", [])
    assert len(history) == 1
    assert "HTTP 503" in history[0][1]
    assert "HTTP 503" in sources


def test_clear_chat():
    import frontend.app as frontend

    msg, history, sources = frontend.clear_chat()
    assert msg == ""
    assert history == []
    assert sources == ""


def test_scope_choices():
    import frontend.app as frontend

    choices = [("file1.pdf", "doc-1"), ("file2.pdf", "doc-2")]
    scoped = frontend._scope_choices(choices)
    assert scoped[0] == ("All Documents", "")
    assert scoped[1:] == choices


def test_refresh_ui(monkeypatch):
    import frontend.app as frontend

    monkeypatch.setattr(
        frontend.httpx,
        "get",
        lambda *args, **kwargs: _response(
            [{"filename": "doc.pdf", "document_id": "doc-1", "chunks": 5}]
        ),
    )
    status, doc_update, scope_update = frontend.refresh_ui()
    assert "Loaded 1" in status
    assert doc_update["value"] == "doc-1"
    assert scope_update["value"] == ""
    assert len(scope_update["choices"]) == 2  # All Documents + doc.pdf


def test_chat_stream_success(monkeypatch):
    import frontend.app as frontend
    from contextlib import contextmanager

    posted_payload = {}

    @contextmanager
    def fake_stream(method, url, json=None, **kwargs):
        nonlocal posted_payload
        posted_payload = json or {}
        resp = MagicMock(spec=httpx.Response)
        resp.iter_lines.return_value = [
            'event: sources',
            'data: {"sources": [{"filename": "doc.pdf", "page": 1, "score": 0.85}]}',
            '',
            'event: token',
            'data: {"token": "The answer "}',
            '',
            'event: token',
            'data: {"token": "is 42."}',
            '',
            'event: done',
            'data: {}',
        ]
        resp.raise_for_status = MagicMock()
        yield resp

    from unittest.mock import MagicMock
    monkeypatch.setattr(frontend.httpx, "stream", fake_stream)

    updates = list(frontend.chat_stream("What is answer?", [], model="llama3.2:1b", selected_doc_id="doc-1"))
    assert len(updates) >= 3
    final_msg, final_hist, final_sources = updates[-1]
    assert final_msg == ""
    assert final_hist[-1][0] == "What is answer?"
    assert final_hist[-1][1] == "The answer is 42."
    assert "doc.pdf" in final_sources
    assert "Page 1" in final_sources
    assert posted_payload["doc_id"] == "doc-1"
    assert posted_payload["model"] == "llama3.2:1b"


def test_chat_stream_empty_message():
    import frontend.app as frontend

    updates = list(frontend.chat_stream("   ", [["q", "a"]]))
    assert len(updates) == 1
    assert "Please enter a valid question" in updates[0][2]

