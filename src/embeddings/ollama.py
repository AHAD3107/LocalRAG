"""Small HTTP client for local Ollama embeddings and chat generation."""
from __future__ import annotations

import json
from typing import Any, Iterator

import httpx


class OllamaError(RuntimeError):
    """A recoverable Ollama connectivity or response error."""


class OllamaClient:
    def __init__(
        self,
        base_url: str,
        timeout: float = 60.0,
        http_client: httpx.Client | None = None,
        readiness_timeout: float = 3.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.readiness_timeout = readiness_timeout
        self._custom_client = http_client is not None
        self._client: httpx.Client | None = http_client

    def _get_client(self) -> httpx.Client:
        if self._client is None:
            self._client = httpx.Client(timeout=self.timeout)
            self._custom_client = False
        return self._client

    def close(self) -> None:
        """Close the managed HTTP client connection pool if owned."""
        if not self._custom_client and self._client is not None:
            self._client.close()
            self._client = None

    def __enter__(self) -> "OllamaClient":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def _post(self, path: str, payload: dict[str, Any], timeout: float | None = None) -> dict[str, Any]:
        req_timeout = self.timeout if timeout is None else timeout
        client = self._get_client()
        try:
            response = client.post(self.base_url + path, json=payload, timeout=req_timeout)
            response.raise_for_status()
            data = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise OllamaError("Ollama is unavailable or returned an invalid response.") from exc
        if not isinstance(data, dict):
            raise OllamaError("Ollama returned an invalid response.")
        return data

    def embed(self, text: str, model: str) -> list[float]:
        data = self._post("/api/embeddings", {"model": model, "prompt": text})
        embedding = data.get("embedding")
        if not isinstance(embedding, list) or not embedding or not all(isinstance(v, (int, float)) for v in embedding):
            raise OllamaError("Ollama returned no valid embedding.")
        return [float(v) for v in embedding]

    def embed_batch(self, texts: list[str], model: str, batch_size: int = 32) -> list[list[float]]:
        if not texts:
            return []
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

        all_embeddings: list[list[float]] = []
        client = self._get_client()

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            batch_embeddings: list[list[float]] | None = None

            # Attempt modern /api/embed batch endpoint first
            try:
                response = client.post(
                    self.base_url + "/api/embed",
                    json={"model": model, "input": batch},
                    timeout=self.timeout,
                )
                if response.is_success:
                    data = response.json()
                    raw_embeddings = data.get("embeddings")
                    if (
                        isinstance(raw_embeddings, list)
                        and len(raw_embeddings) == len(batch)
                        and all(
                            isinstance(v, list)
                            and v
                            and all(isinstance(x, (int, float)) for x in v)
                            for v in raw_embeddings
                        )
                    ):
                        dim = len(raw_embeddings[0])
                        if all(len(v) == dim for v in raw_embeddings):
                            batch_embeddings = [[float(x) for x in v] for v in raw_embeddings]
            except (httpx.HTTPError, ValueError):
                batch_embeddings = None

            # Fallback to individual embed calls if /api/embed failed or is unsupported
            if batch_embeddings is None:
                batch_embeddings = [self.embed(t, model) for t in batch]

            all_embeddings.extend(batch_embeddings)

        # Validate uniform dimensions across all batches
        if all_embeddings:
            expected_dim = len(all_embeddings[0])
            if any(len(v) != expected_dim for v in all_embeddings):
                raise OllamaError("Inconsistent embedding dimensions returned across batches.")

        return all_embeddings

    def chat(self, prompt: str, model: str) -> str:
        data = self._post(
            "/api/generate",
            {"model": model, "prompt": prompt, "stream": False},
        )
        response = data.get("response")
        if not isinstance(response, str) or not response.strip():
            raise OllamaError("Ollama returned no generated answer.")
        return response.strip()

    def chat_stream(self, prompt: str, model: str) -> Iterator[str]:
        client = self._get_client()
        try:
            with client.stream(
                "POST",
                self.base_url + "/api/generate",
                json={"model": model, "prompt": prompt, "stream": True},
                timeout=self.timeout,
            ) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        token = data.get("response", "")
                        if token:
                            yield token
                        if data.get("done", False):
                            break
                    except (json.JSONDecodeError, ValueError):
                        continue
        except (httpx.HTTPError, ValueError) as exc:
            raise OllamaError("Ollama streaming error or connection lost.") from exc

    def is_available(self, timeout: float | None = None) -> bool:
        check_timeout = self.readiness_timeout if timeout is None else timeout
        client = self._get_client()
        try:
            response = client.get(self.base_url, timeout=check_timeout)
            return response.is_success
        except (httpx.HTTPError, ValueError):
            return False
