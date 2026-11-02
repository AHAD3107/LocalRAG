"""Environment-backed configuration for LocalRAG."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent


def _path(value: str) -> Path:
    candidate = Path(value)
    return candidate if candidate.is_absolute() else BASE_DIR / candidate


def _csv(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


@dataclass(frozen=True)
class Settings:
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
    chat_model_default: str = os.getenv("CHAT_MODEL_DEFAULT", "llama3.2:1b")
    chat_model_advanced: str = os.getenv("CHAT_MODEL_ADVANCED", "llama3:8b")
    documents_dir: Path = _path(os.getenv("DOCUMENTS_DIR", "data/documents"))
    index_dir: Path = _path(os.getenv("INDEX_DIR", "data/index"))
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "500"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "50"))
    top_k: int = int(os.getenv("TOP_K", "4"))
    similarity_threshold: float = float(os.getenv("SIMILARITY_THRESHOLD", "0.55"))
    retrieval_strategy: str = os.getenv("RETRIEVAL_STRATEGY", "hybrid_rerank")
    reranker_model: str = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
    reranker_score_threshold: float = float(os.getenv("RERANKER_SCORE_THRESHOLD", "-4.0"))
    reranker_first_stage_depth: int = int(os.getenv("RERANKER_FIRST_STAGE_DEPTH", "10"))
    reranker_device: str = os.getenv("RERANKER_DEVICE", "cpu")
    api_host: str = os.getenv("API_HOST", "127.0.0.1")
    api_port: int = int(os.getenv("API_PORT", "8000"))
    frontend_port: int = int(os.getenv("FRONTEND_PORT", "7860"))
    max_upload_bytes: int = int(os.getenv("MAX_UPLOAD_BYTES", str(10 * 1024 * 1024)))
    request_timeout_seconds: float = float(os.getenv("REQUEST_TIMEOUT_SECONDS", "60"))
    readiness_timeout_seconds: float = float(os.getenv("READINESS_TIMEOUT_SECONDS", "3.0"))
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    allowed_origins: tuple[str, ...] = _csv(
        os.getenv("ALLOWED_ORIGINS", "http://127.0.0.1:7860,http://localhost:7860")
    )

    def __post_init__(self) -> None:
        valid_strategies = {"hybrid_rerank", "bm25_rerank", "hybrid", "bm25", "dense"}
        if self.retrieval_strategy.lower() not in valid_strategies:
            raise ValueError(
                f"Invalid retrieval_strategy '{self.retrieval_strategy}'. "
                f"Must be one of: {sorted(valid_strategies)}"
            )
        if self.chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {self.chunk_size}")
        if self.chunk_overlap < 0 or self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                f"chunk_overlap must be in [0, chunk_size), got {self.chunk_overlap} (chunk_size={self.chunk_size})"
            )
        if self.top_k <= 0:
            raise ValueError(f"top_k must be positive, got {self.top_k}")

    def ensure_directories(self) -> None:
        self.documents_dir.mkdir(parents=True, exist_ok=True)
        self.index_dir.mkdir(parents=True, exist_ok=True)


settings = Settings()
settings.ensure_directories()

# Backwards-compatible names used by early repository documentation.
DATA_DIR = BASE_DIR / "data"
DOCUMENTS_DIR = settings.documents_dir
INDEX_DIR = settings.index_dir
OLLAMA_BASE_URL = settings.ollama_base_url
EMBEDDING_MODEL = settings.embedding_model
CHAT_MODEL_DEFAULT = settings.chat_model_default
CHAT_MODEL_ADVANCED = settings.chat_model_advanced
DEFAULT_CHUNK_SIZE = settings.chunk_size
DEFAULT_CHUNK_OVERLAP = settings.chunk_overlap
DEFAULT_TOP_K = settings.top_k
SIMILARITY_THRESHOLD = settings.similarity_threshold
API_HOST = settings.api_host
API_PORT = settings.api_port
FRONTEND_PORT = settings.frontend_port
READINESS_TIMEOUT_SECONDS = settings.readiness_timeout_seconds
