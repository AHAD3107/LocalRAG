"""Tests for Ollama client readiness checks, connection pooling, and batching."""
from unittest.mock import MagicMock
import httpx
import pytest

from src.embeddings.ollama import OllamaClient, OllamaError


def test_ollama_is_available_success():
    mock_client = MagicMock(spec=httpx.Client)
    mock_response = MagicMock(spec=httpx.Response)
    mock_response.is_success = True
    mock_client.get.return_value = mock_response

    client = OllamaClient("http://127.0.0.1:11434", timeout=60.0, readiness_timeout=3.0, http_client=mock_client)
    assert client.is_available() is True
    mock_client.get.assert_called_once_with("http://127.0.0.1:11434", timeout=3.0)


def test_ollama_is_available_connection_error():
    mock_client = MagicMock(spec=httpx.Client)
    mock_client.get.side_effect = httpx.ConnectError("Connection refused")

    client = OllamaClient("http://127.0.0.1:11434", timeout=60.0, readiness_timeout=3.0, http_client=mock_client)
    assert client.is_available() is False
    mock_client.get.assert_called_once_with("http://127.0.0.1:11434", timeout=3.0)


def test_ollama_is_available_timeout():
    mock_client = MagicMock(spec=httpx.Client)
    mock_client.get.side_effect = httpx.ReadTimeout("Read timed out")

    client = OllamaClient("http://127.0.0.1:11434", timeout=60.0, readiness_timeout=2.5, http_client=mock_client)
    assert client.is_available() is False
    mock_client.get.assert_called_once_with("http://127.0.0.1:11434", timeout=2.5)


def test_ollama_operations_retain_general_timeout():
    mock_client = MagicMock(spec=httpx.Client)
    mock_response = MagicMock(spec=httpx.Response)
    mock_response.json.return_value = {"embedding": [0.1, 0.2, 0.3]}
    mock_client.post.return_value = mock_response

    client = OllamaClient("http://127.0.0.1:11434", timeout=45.0, readiness_timeout=3.0, http_client=mock_client)
    result = client.embed("test text", "nomic-embed-text")
    assert result == [0.1, 0.2, 0.3]
    mock_client.post.assert_called_once_with(
        "http://127.0.0.1:11434/api/embeddings",
        json={"model": "nomic-embed-text", "prompt": "test text"},
        timeout=45.0,
    )


def test_ollama_client_connection_reuse_and_cleanup():
    client = OllamaClient("http://127.0.0.1:11434", timeout=10.0)
    session1 = client._get_client()
    session2 = client._get_client()
    assert session1 is session2
    assert not session1.is_closed

    client.close()
    assert session1.is_closed
    assert client._client is None


def test_ollama_client_context_manager():
    with OllamaClient("http://127.0.0.1:11434", timeout=10.0) as client:
        session = client._get_client()
        assert not session.is_closed
    assert session.is_closed


def test_ollama_embed_batch_success():
    mock_client = MagicMock(spec=httpx.Client)
    mock_response = MagicMock(spec=httpx.Response)
    mock_response.is_success = True
    mock_response.json.return_value = {
        "embeddings": [
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
        ]
    }
    mock_client.post.return_value = mock_response

    client = OllamaClient("http://127.0.0.1:11434", timeout=30.0, http_client=mock_client)
    vectors = client.embed_batch(["text 1", "text 2"], "nomic-embed-text")
    assert vectors == [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
    mock_client.post.assert_called_once_with(
        "http://127.0.0.1:11434/api/embed",
        json={"model": "nomic-embed-text", "input": ["text 1", "text 2"]},
        timeout=30.0,
    )


def test_ollama_embed_batch_fallback_on_unsupported_endpoint():
    mock_client = MagicMock(spec=httpx.Client)

    def fake_post(url, **kwargs):
        if url.endswith("/api/embed"):
            resp = MagicMock(spec=httpx.Response)
            resp.is_success = False
            resp.status_code = 404
            return resp
        elif url.endswith("/api/embeddings"):
            resp = MagicMock(spec=httpx.Response)
            resp.is_success = True
            prompt = kwargs.get("json", {}).get("prompt", "")
            resp.json.return_value = {"embedding": [1.0, 2.0] if "first" in prompt else [3.0, 4.0]}
            return resp
        raise ValueError(f"Unexpected url {url}")

    mock_client.post.side_effect = fake_post
    client = OllamaClient("http://127.0.0.1:11434", timeout=30.0, http_client=mock_client)
    vectors = client.embed_batch(["first chunk", "second chunk"], "nomic-embed-text")
    assert vectors == [[1.0, 2.0], [3.0, 4.0]]


def test_ollama_embed_batch_empty_input():
    client = OllamaClient("http://127.0.0.1:11434")
    assert client.embed_batch([], "nomic-embed-text") == []


def test_ollama_chat_stream_success():
    from contextlib import contextmanager

    mock_client = MagicMock(spec=httpx.Client)
    mock_resp = MagicMock(spec=httpx.Response)
    mock_resp.iter_lines.return_value = [
        '{"response": "Hello ", "done": false}',
        '',
        '{"response": "world!", "done": true}',
    ]

    @contextmanager
    def fake_stream(*args, **kwargs):
        yield mock_resp

    mock_client.stream = fake_stream
    client = OllamaClient("http://127.0.0.1:11434", timeout=30.0, http_client=mock_client)
    tokens = list(client.chat_stream("Say hi", "llama3.2:1b"))
    assert tokens == ["Hello ", "world!"]


def test_ollama_chat_stream_error():
    from contextlib import contextmanager

    mock_client = MagicMock(spec=httpx.Client)
    mock_resp = MagicMock(spec=httpx.Response)
    mock_resp.raise_for_status.side_effect = httpx.HTTPStatusError("500", request=MagicMock(), response=MagicMock())

    @contextmanager
    def fake_stream(*args, **kwargs):
        yield mock_resp

    mock_client.stream = fake_stream
    client = OllamaClient("http://127.0.0.1:11434", timeout=30.0, http_client=mock_client)
    with pytest.raises(OllamaError, match="streaming error"):
        list(client.chat_stream("Say hi", "llama3.2:1b"))

