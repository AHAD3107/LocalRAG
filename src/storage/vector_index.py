"""Persistent NumPy vector index with atomic writes."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import zipfile

import numpy as np

from src.ingestion.models import Chunk


class IndexError(RuntimeError):
    """Raised when the persisted index is missing or incompatible."""


class VectorIndex:
    def __init__(
        self,
        directory: Path,
        embedding_model: str,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
    ):
        self.directory = directory
        self.embedding_model = embedding_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.vectors = np.empty((0, 0), dtype=np.float32)
        self.chunks: list[Chunk] = []

    @property
    def path(self) -> Path:
        return self.directory / "index.npz"

    def add(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        if not chunks:
            return
        if any(existing.doc_id == chunks[0].doc_id for existing in self.chunks):
            raise ValueError("document is already indexed")
        if any(chunk.doc_id != chunks[0].doc_id for chunk in chunks):
            raise ValueError("a single index operation must contain one document")
        matrix = np.asarray(vectors, dtype=np.float32)
        if matrix.ndim != 2 or matrix.shape[0] != len(chunks):
            raise ValueError("vector count does not match chunk count")
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        if np.any(norms == 0):
            raise ValueError("zero-length embeddings are not indexable")
        matrix /= norms
        if self.vectors.size and matrix.shape[1] != self.vectors.shape[1]:
            raise ValueError("embedding dimension does not match existing index")
        self.vectors = matrix if not self.chunks else np.vstack((self.vectors, matrix))
        self.chunks.extend(chunks)

    def remove_document(self, doc_id: str) -> int:
        keep = [i for i, chunk in enumerate(self.chunks) if chunk.doc_id != doc_id]
        removed = len(self.chunks) - len(keep)
        if not removed:
            return 0
        self.chunks = [self.chunks[i] for i in keep]
        self.vectors = self.vectors[keep] if keep else np.empty((0, self.vectors.shape[1]), dtype=np.float32)
        return removed

    def save(self) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        metadata = {
            "embedding_model": self.embedding_model,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "dimension": int(self.vectors.shape[1]) if self.vectors.ndim == 2 else 0,
            "chunks": [
                {
                    "chunk_id": chunk.chunk_id,
                    "doc_id": chunk.doc_id,
                    "source_filename": chunk.source_filename,
                    "page_number": chunk.page_number,
                    "chunk_index": chunk.chunk_index,
                    "text": chunk.text,
                }
                for chunk in self.chunks
            ],
        }
        metadata_json = json.dumps(metadata, separators=(",", ":"))
        metadata_path: Path | None = None
        vector_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(dir=self.directory, suffix=".npz", delete=False) as handle:
                vector_path = Path(handle.name)
            with tempfile.NamedTemporaryFile(
                dir=self.directory,
                suffix=".json",
                delete=False,
                mode="w",
                encoding="utf-8",
            ) as handle:
                metadata_path = Path(handle.name)
                handle.write(metadata_json)
            np.savez_compressed(vector_path, vectors=self.vectors, metadata=np.array(metadata_json))
            vector_path.replace(self.path)
            metadata_path.replace(self.directory / "index.json")
        finally:
            if vector_path is not None:
                vector_path.unlink(missing_ok=True)
            if metadata_path is not None:
                metadata_path.unlink(missing_ok=True)

    @classmethod
    def load(
        cls,
        directory: Path,
        embedding_model: str,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
    ) -> "VectorIndex":
        metadata_path = directory / "index.json"
        vector_path = directory / "index.npz"
        if not metadata_path.exists() and not vector_path.exists():
            return cls(directory, embedding_model, chunk_size, chunk_overlap)
        if not metadata_path.exists() or not vector_path.exists():
            raise IndexError("persisted vector index is incomplete")
        try:
            with np.load(vector_path, allow_pickle=False) as archive:
                vectors = np.array(archive["vectors"], copy=True)
                embedded_metadata = archive["metadata"].item()
            metadata = json.loads(str(embedded_metadata))
            sidecar_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if sidecar_metadata != metadata:
                raise IndexError("persisted vector index metadata is inconsistent")
            if metadata.get("embedding_model") != embedding_model:
                raise IndexError("index embedding model does not match configured model")
            if metadata.get("chunk_size") != chunk_size or metadata.get("chunk_overlap") != chunk_overlap:
                raise IndexError("index chunking configuration does not match configured values")
            chunks = [Chunk(**item) for item in metadata.get("chunks", [])]
        except (
            OSError,
            ValueError,
            KeyError,
            TypeError,
            AttributeError,
            json.JSONDecodeError,
            zipfile.BadZipFile,
        ) as exc:
            raise IndexError("persisted vector index is invalid") from exc
        if vectors.ndim != 2 or vectors.shape[0] != len(chunks):
            raise IndexError("persisted vector index metadata is inconsistent")
        dimension = metadata.get("dimension")
        if dimension != (int(vectors.shape[1]) if vectors.ndim == 2 else None):
            raise IndexError("persisted vector index dimension metadata is inconsistent")
        index = cls(directory, embedding_model, chunk_size, chunk_overlap)
        index.vectors, index.chunks = vectors, chunks
        return index
