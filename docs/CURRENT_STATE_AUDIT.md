# Current State Audit

## Evidence-based baseline

The repository was inspected before implementation. The only executable behavior
present at the baseline was the PyMuPDF ingestion foundation:

- `src/ingestion/models.py` defines validated page, document, chunk, and result
  dataclasses.
- `src/ingestion/pdf_extractor.py` reads PDFs with PyMuPDF, preserves one-based
  page metadata, normalizes text, detects empty/encrypted/corrupt documents, and
  closes document handles.
- `src/ingestion/pdf_loader.py` is a compatibility re-export.
- The baseline suite contained 43 passing tests covering those modules.

The baseline test command was:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

## Missing or unverified capabilities

At baseline there were no implementation modules or entry points for:

- text chunking;
- Ollama embeddings or chat generation;
- persistent vector storage;
- semantic retrieval;
- FastAPI routes;
- a Gradio application;
- health/readiness checks;
- Docker or Compose deployment;
- CI automation.

The README described these as intended features and explicitly stated that the
application was not yet runnable. The existing `config.py` created data
directories at import time and hard-coded the Ollama URL, model settings, and
server values. No `.env.example` existed.

## Initial architecture

The intended architecture was a FastAPI backend plus Gradio frontend, but that
boundary was not implemented. `frontend/` contained only an empty package marker.
The data directories contained only `.gitkeep` files and no runtime index.

## Risks found

- No upload validation or size limit existed.
- No safe filename or overwrite policy existed.
- No persistence format or index compatibility metadata existed.
- No configured timeout or useful upstream error handling existed.
- Public deployment would have had no API authentication or restrictive CORS.
- The repository had no executable launch command for an end-to-end workflow.

## Scope of the completed work

The implementation adds a lightweight local stack using the existing dependencies:
PyMuPDF, NumPy, HTTPX, FastAPI, Pydantic, and Gradio. It does not add LangChain,
a hosted vector database, or a cloud AI provider.
