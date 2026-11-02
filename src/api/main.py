"""FastAPI application for LocalRAG."""
from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
import re
from typing import Annotated, Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from fastapi.responses import JSONResponse, StreamingResponse

from config import settings
from src.embeddings.ollama import OllamaClient, OllamaError
from src.generation.contextualize import contextualize_query
from src.generation.prompt import build_prompt
from src.ingestion.pdf_extractor import extract_pdf_result
from src.ingestion.models import Document
from src.ingestion.text_splitter import chunk_document
from src.retrieval.pipeline import retrieve_context
from src.retrieval.search import search
from src.storage.vector_index import VectorIndex, IndexError

logger = logging.getLogger(__name__)
app = FastAPI(title="LocalRAG API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=list(settings.allowed_origins),
    allow_credentials=False,
    allow_methods=["DELETE", "GET", "POST"],
    allow_headers=["*"],
)

_index_lock = asyncio.Lock()


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    model: str | None = Field(default=None, min_length=1, max_length=100)
    doc_id: str | None = Field(default=None, min_length=1, max_length=100)
    history: list[Any] | None = Field(default=None)


class Source(BaseModel):
    filename: str
    page: int
    score: float
    text: str


class QueryResponse(BaseModel):
    answer: str
    sources: list[Source]


def _index() -> VectorIndex:
    try:
        return VectorIndex.load(
            settings.index_dir,
            settings.embedding_model,
            settings.chunk_size,
            settings.chunk_overlap,
        )
    except IndexError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


def _safe_name(filename: str) -> str:
    name = Path(filename).name
    name = re.sub(r"[^A-Za-z0-9._-]", "_", name)
    if not name.lower().endswith(".pdf") or name in {"", ".pdf"}:
        raise HTTPException(status_code=415, detail="Only PDF uploads are supported.")
    return name


def _meaningful_terms(text: str) -> set[str]:
    return {
        term
        for term in re.findall(r"[a-z0-9]{3,}", text.lower())
        if term not in {"the", "what", "which", "where", "when", "does", "this", "that", "about"}
    }


def _looks_like_refusal(answer: str) -> bool:
    lowered = answer.lower()
    return any(
        phrase in lowered
        for phrase in ("not enough evidence", "insufficient evidence", "no direct", "cannot answer")
    )


def _sync_index_document(filename: str, doc_id: str, pages: list) -> int:
    client = OllamaClient(settings.ollama_base_url, settings.request_timeout_seconds)
    try:
        document = Document(
            doc_id=doc_id, source_filename=filename, pages=pages
        )
        chunks = chunk_document(document, settings.chunk_size, settings.chunk_overlap)
        if hasattr(client, "embed_batch"):
            vectors = client.embed_batch([chunk.text for chunk in chunks], settings.embedding_model)
        else:
            vectors = [client.embed(chunk.text, settings.embedding_model) for chunk in chunks]
        index = _index()
        index.add(chunks, vectors)
        index.save()
        return len(chunks)
    finally:
        if hasattr(client, "close"):
            client.close()


def _sync_delete_document(doc_id: str) -> int:
    index = _index()
    removed = index.remove_document(doc_id)
    if not removed:
        raise HTTPException(status_code=404, detail="Document was not found in the index.")
    index.save()
    for path in settings.documents_dir.iterdir():
        if path.is_file() and path.suffix.lower() == ".pdf":
            result = extract_pdf_result(path)
            if result.doc_id == doc_id:
                path.unlink(missing_ok=True)
                break
    return removed


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/ready", response_model=None)
async def ready() -> dict[str, object] | JSONResponse:
    client = OllamaClient(
        settings.ollama_base_url,
        settings.request_timeout_seconds,
        readiness_timeout=settings.readiness_timeout_seconds,
    )
    try:
        index = await asyncio.to_thread(_index)
        available = await asyncio.to_thread(client.is_available)
        payload = {"status": "ready" if available else "degraded", "ollama": available, "indexed_chunks": len(index.chunks)}
        if not available:
            return JSONResponse(status_code=503, content=payload)
        return payload
    finally:
        if hasattr(client, "close"):
            client.close()


@app.get("/documents")
async def documents() -> list[dict[str, object]]:
    index = await asyncio.to_thread(_index)
    docs: dict[str, dict[str, object]] = {}
    for chunk in index.chunks:
        docs.setdefault(chunk.doc_id, {"document_id": chunk.doc_id, "filename": chunk.source_filename, "chunks": 0})
        docs[chunk.doc_id]["chunks"] = int(docs[chunk.doc_id]["chunks"]) + 1
    return list(docs.values())


@app.delete("/documents/{doc_id}")
async def delete_document(doc_id: str) -> dict[str, int]:
    async with _index_lock:
        removed = await asyncio.to_thread(_sync_delete_document, doc_id)
    return {"removed_chunks": removed}


@app.post("/upload")
async def upload(file: Annotated[UploadFile, File(...)]) -> dict[str, object]:
    filename = _safe_name(file.filename or "")
    target = settings.documents_dir / filename
    if target.exists():
        raise HTTPException(status_code=409, detail="A document with that filename already exists.")
    content = await file.read(settings.max_upload_bytes + 1)
    if len(content) > settings.max_upload_bytes:
        raise HTTPException(status_code=413, detail="Uploaded file exceeds the configured size limit.")
    if not content.startswith(b"%PDF"):
        raise HTTPException(status_code=415, detail="Uploaded content is not a PDF.")

    temporary = settings.documents_dir / f".{filename}.uploading"
    temporary.write_bytes(content)
    result = extract_pdf_result(temporary)
    if not result.is_usable:
        temporary.unlink(missing_ok=True)
        raise HTTPException(status_code=422, detail=result.error_message or "; ".join(result.warnings) or "PDF has no extractable text.")

    try:
        async with _index_lock:
            chunk_count = await asyncio.to_thread(
                _sync_index_document, filename, result.doc_id or "", result.pages
            )
    except (OllamaError, ValueError, OSError) as exc:
        temporary.unlink(missing_ok=True)
        logger.warning("Upload indexing failed for %s: %s", filename, type(exc).__name__)
        if isinstance(exc, ValueError) and str(exc) == "document is already indexed":
            raise HTTPException(status_code=409, detail="This document is already indexed.") from exc
        raise HTTPException(status_code=503, detail="Document could not be indexed. Check Ollama and configuration.") from exc
    temporary.replace(target)
    return {"document_id": result.doc_id, "filename": filename, "status": result.status.value, "chunks": chunk_count, "warnings": result.warnings}


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest) -> QueryResponse:
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=422, detail="Question must contain non-whitespace text.")
    client = OllamaClient(settings.ollama_base_url, settings.request_timeout_seconds)
    model = request.model or settings.chat_model_default
    try:
        index = _index()
        search_query = question
        if request.history:
            search_query = contextualize_query(question, request.history, client, model)

        logger.info(
            "Executing query: '%s' | Strategy: %s (doc_id=%s)",
            question[:60],
            settings.retrieval_strategy,
            request.doc_id,
        )
        matches = retrieve_context(
            query=search_query,
            index=index,
            client=client,
            top_k=settings.top_k,
            doc_id=request.doc_id,
        )
        if not matches:
            return QueryResponse(answer="I could not find enough evidence in the indexed documents to answer that question.", sources=[])
        answer = client.chat(build_prompt(question, matches), model)
        source_terms = _meaningful_terms(matches[0][0].text)
        question_overlap = len(_meaningful_terms(question) & source_terms)
        conf_threshold = (
            settings.similarity_threshold
            if settings.retrieval_strategy == "dense"
            else settings.reranker_score_threshold
        )
        if matches[0][1] >= conf_threshold and question_overlap >= 2 and _looks_like_refusal(answer):
            answer = client.chat(build_prompt(question, matches), model)
            if _looks_like_refusal(answer):
                answer = f"{matches[0][0].text} [Source 1]"
        answer = re.sub(r"\[Source\s+(\d+)\]", lambda match: match.group(0) if int(match.group(1)) <= len(matches) else "", answer)
        return QueryResponse(
            answer=answer,
            sources=[Source(filename=chunk.source_filename, page=chunk.page_number, score=score, text=chunk.text) for chunk, score in matches],
        )
    except ValueError as exc:
        if "not found in the index" in str(exc):
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        raise HTTPException(status_code=503, detail="The local AI service could not answer this request.") from exc
    except OllamaError as exc:
        raise HTTPException(status_code=503, detail="The local AI service could not answer this request.") from exc
    finally:
        if hasattr(client, "close"):
            client.close()


@app.post("/query/stream")
def query_stream(request: QueryRequest) -> StreamingResponse:
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=422, detail="Question must contain non-whitespace text.")
    model = request.model or settings.chat_model_default

    def event_generator():
        client = OllamaClient(settings.ollama_base_url, settings.request_timeout_seconds)
        try:
            index = _index()
            search_query = question
            if request.history:
                search_query = contextualize_query(question, request.history, client, model)

            logger.info(
                "Streaming query: '%s' | Strategy: %s (doc_id=%s)",
                question[:60],
                settings.retrieval_strategy,
                request.doc_id,
            )
            matches = retrieve_context(
                query=search_query,
                index=index,
                client=client,
                top_k=settings.top_k,
                doc_id=request.doc_id,
            )

            if not matches:
                yield f"event: sources\ndata: {json.dumps({'sources': []})}\n\n"
                refusal_msg = "I could not find enough evidence in the indexed documents to answer that question."
                yield f"event: token\ndata: {json.dumps({'token': refusal_msg})}\n\n"
                yield f"event: done\ndata: {json.dumps({})}\n\n"
                return

            sources_payload = [
                {"filename": chunk.source_filename, "page": chunk.page_number, "score": score, "text": chunk.text}
                for chunk, score in matches
            ]
            yield f"event: sources\ndata: {json.dumps({'sources': sources_payload})}\n\n"

            prompt = build_prompt(question, matches)
            if hasattr(client, "chat_stream"):
                for token in client.chat_stream(prompt, model):
                    yield f"event: token\ndata: {json.dumps({'token': token})}\n\n"
            else:
                full_resp = client.chat(prompt, model)
                yield f"event: token\ndata: {json.dumps({'token': full_resp})}\n\n"

            yield f"event: done\ndata: {json.dumps({})}\n\n"
        except ValueError as exc:
            if "not found in the index" in str(exc):
                yield f"event: error\ndata: {json.dumps({'error': str(exc)})}\n\n"
            else:
                yield f"event: error\ndata: {json.dumps({'error': 'The local AI service could not answer this request.'})}\n\n"
        except OllamaError:
            yield f"event: error\ndata: {json.dumps({'error': 'The local AI service could not answer this request.'})}\n\n"
        finally:
            if hasattr(client, "close"):
                client.close()

    return StreamingResponse(event_generator(), media_type="text/event-stream")
