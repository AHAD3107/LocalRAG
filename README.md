# Local RAG Project

A fully local, privacy-first Retrieval-Augmented Generation (RAG) system for question-answering over PDF documents.

## Technology Stack
- **Backend API:** FastAPI
- **Frontend GUI:** Gradio (HTTP-only client)
- **Local LLM & Embeddings:** Ollama (`nomic-embed-text`, `llama3.2:1b`, `llama3:8b`)
- **Vector Storage:** Persistent NumPy vector index with cosine similarity
- **PDF Extraction:** PyMuPDF (`fitz`)
- **Validation:** Pydantic
- **Testing:** pytest
