# Implementation Plan

1. Establish environment-backed configuration and protected runtime directories.
2. Add deterministic, page-aware chunking with validation and tests.
3. Add an Ollama client with timeouts, response validation, and clear errors.
4. Add persistent atomic vector indexing and cosine retrieval.
5. Connect ingestion, indexing, retrieval, and grounded generation in an API.
6. Add a real Gradio client, Docker Compose files, health checks, and CI.
7. Update documentation and run the complete available test suite.

## Acceptance evidence

- Unit tests cover malformed/empty PDFs, chunk boundaries, embedding failures,
  index compatibility, retrieval ranking, and API validation.
- An end-to-end test uses mocked Ollama responses and verifies page attribution.
- A missing-answer test verifies the API does not generate without evidence.
- Docker and local commands are documented only after matching files exist.
