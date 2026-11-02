# LocalRAG Architecture

## Runtime flow

```text
PDF multipart upload
  -> API validation and safe storage
  -> PyMuPDF page extraction
  -> page-bounded character chunks
  -> Ollama embedding requests
  -> atomic NumPy index persistence

Question
  -> Ollama query embedding
  -> cosine similarity search
  -> thresholded source chunks
  -> evidence-constrained Ollama chat prompt
  -> answer and page-level sources
```

The index stores vectors and JSON metadata together with the embedding model and
chunking settings. Loading rejects incompatible configuration rather than
silently mixing vectors from different models. The default retrieval threshold
is 0.70; it should be tuned only against representative local documents.

## Components

- `config.py`: environment-backed settings and runtime directories.
- `src/ingestion`: PDF validation, extraction, normalization, and chunking.
- `src/embeddings`: Ollama HTTP client and embedding response validation.
- `src/storage`: atomic persistent NumPy/JSON vector index.
- `src/retrieval`: cosine ranking and threshold filtering.
- `src/generation`: evidence-only chat prompt and Ollama generation client.
- `src/api`: FastAPI application, upload/query schemas, health/readiness, and
  dependency wiring.
- `frontend/app.py`: Gradio client of the real API.

## Persistence

Uploaded PDFs are stored under `DOCUMENTS_DIR`; the index is stored under
`INDEX_DIR`. Both locations are configurable and should be mounted as persistent
storage in containers. Index replacement uses a temporary file followed by an
atomic replace. Runtime files are excluded from Git.

## Privacy boundary

Document text, embeddings, questions, and answers are sent only to the configured
Ollama endpoint. If that endpoint is remote, the data boundary includes that
network and host. The default is loopback for local development. Public exposure
requires authentication, HTTPS, restrictive CORS, and a separately secured
Ollama service; Compose does not provide those controls by itself.
