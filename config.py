"""
Local RAG Application Configuration
"""
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DOCUMENTS_DIR = DATA_DIR / "documents"
INDEX_DIR = DATA_DIR / "index"

# Ensure runtime directories exist
DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
INDEX_DIR.mkdir(parents=True, exist_ok=True)

# Ollama Settings
OLLAMA_BASE_URL = "http://127.0.0.1:11434"
EMBEDDING_MODEL = "nomic-embed-text"
CHAT_MODEL_DEFAULT = "llama3.2:1b"
CHAT_MODEL_ADVANCED = "llama3:8b"

# Chunking Configuration
DEFAULT_CHUNK_SIZE = 500       # Target characters per chunk
DEFAULT_CHUNK_OVERLAP = 50     # Overlap characters between consecutive chunks

# Retrieval Configuration
DEFAULT_TOP_K = 4
SIMILARITY_THRESHOLD = 0.35    # Minimum cosine similarity threshold

# Server Settings
API_HOST = "127.0.0.1"
API_PORT = 8000
FRONTEND_PORT = 7860
