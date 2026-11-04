"""Gradio frontend that calls the LocalRAG API."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Iterator

# Ensure project root is in sys.path for direct script execution
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import gradio as gr
import gradio_client.utils as _client_utils
import httpx

# Patch gradio_client bug where boolean JSON schemas (e.g. additionalProperties: True) cause TypeError in api_info
_orig_json_schema_to_python_type = _client_utils._json_schema_to_python_type


def _safe_json_schema_to_python_type(schema, defs):
    if isinstance(schema, bool):
        return "bool"
    return _orig_json_schema_to_python_type(schema, defs)


_client_utils._json_schema_to_python_type = _safe_json_schema_to_python_type

from src.monitoring.system_monitor import format_metrics_for_display


def get_api_url() -> str:
    """Resolve API URL with environment precedence: API_URL > http://{API_HOST}:{API_PORT} > http://127.0.0.1:8000."""
    url = os.getenv("API_URL")
    if url and url.strip():
        return url.strip().rstrip("/")
    host = os.getenv("API_HOST", "127.0.0.1").strip()
    port = os.getenv("API_PORT", "8000").strip()
    return f"http://{host}:{port}".rstrip("/")


API_URL = get_api_url()
DEFAULT_CHAT_MODEL = os.getenv("CHAT_MODEL_DEFAULT", "llama3.2:1b")
ADVANCED_CHAT_MODEL = os.getenv("CHAT_MODEL_ADVANCED", "llama3:8b")
AVAILABLE_MODELS = [DEFAULT_CHAT_MODEL, ADVANCED_CHAT_MODEL]

DocumentChoice = tuple[str, str]


def _document_choices() -> tuple[list[DocumentChoice], str | None]:
    try:
        response = httpx.get(f"{API_URL}/documents", timeout=2.0)
        response.raise_for_status()
        payload = response.json()
        if isinstance(payload, dict):
            payload = payload.get("value")
        if not isinstance(payload, list):
            return [], "Document service returned an invalid response."
        choices: list[DocumentChoice] = []
        for item in payload:
            if not isinstance(item, dict):
                return [], "Document service returned an invalid document entry."
            filename = item.get("filename")
            document_id = item.get("document_id")
            if not isinstance(filename, str) or not isinstance(document_id, str):
                return [], "Document service returned an invalid document entry."
            choices.append((filename, document_id))
        return choices, None
    except httpx.HTTPStatusError as exc:
        return [], f"Could not load documents (HTTP {exc.response.status_code})."
    except (httpx.HTTPError, ValueError) as exc:
        return [], f"Could not load documents: {exc}. Ensure FastAPI backend is running at {API_URL}."


def _dropdown_update(
    choices: list[DocumentChoice],
    selected_value: str | None = None,
) -> dict[str, object]:
    valid_values = {value for _, value in choices}
    if selected_value not in valid_values:
        selected_value = choices[0][1] if choices else None
    return gr.update(choices=choices, value=selected_value, type="value")


def _scope_choices(choices: list[DocumentChoice]) -> list[tuple[str, str]]:
    return [("All Documents", "")] + [(name, doc_id) for name, doc_id in choices]


def list_documents() -> list[DocumentChoice]:
    """Return valid label/value choices for the initial dropdown state."""
    choices, _ = _document_choices()
    return choices


def refresh_documents() -> tuple[str, dict[str, object]]:
    choices, error = _document_choices()
    if error:
        return error, _dropdown_update([])
    if not choices:
        return "No indexed documents.", _dropdown_update([])
    return f"Loaded {len(choices)} indexed document(s).", _dropdown_update(choices)


def refresh_ui() -> tuple[str, dict[str, object], dict[str, object]]:
    status, doc_update = refresh_documents()
    choices, _ = _document_choices()
    scope_update = gr.update(choices=_scope_choices(choices), value="")
    return status, doc_update, scope_update


def upload_pdf(file: str | None) -> tuple[str, dict[str, object]]:
    if not file:
        status, update = refresh_documents()
        return f"Choose a PDF first. {status}", update
    try:
        with open(file, "rb") as handle:
            response = httpx.post(
                f"{API_URL}/upload",
                files={"file": (os.path.basename(file), handle, "application/pdf")},
                timeout=120,
            )
        response.raise_for_status()
        data = response.json()
        refresh_status, update = refresh_documents()
        return f"Indexed {data['filename']} ({data['chunks']} chunks). {refresh_status}", update
    except httpx.HTTPStatusError as exc:
        refresh_status, update = refresh_documents()
        if exc.response.status_code == 409:
            return (
                "This document is already indexed. Select it from the indexed collection instead of uploading it again. "
                f"{refresh_status}",
                update,
            )
        detail = ""
        try:
            body = exc.response.json()
            if isinstance(body, dict) and "detail" in body:
                detail = f": {body['detail']}"
        except Exception:
            pass
        return f"Upload failed (HTTP {exc.response.status_code}{detail}). {refresh_status}", update
    except (OSError, httpx.HTTPError, ValueError) as exc:
        refresh_status, update = refresh_documents()
        return f"Upload failed: {exc}. {refresh_status}", update


def upload_ui(file: str | None) -> tuple[str, dict[str, object], dict[str, object]]:
    status, doc_update = upload_pdf(file)
    choices, _ = _document_choices()
    scope_update = gr.update(choices=_scope_choices(choices), value="")
    return status, doc_update, scope_update


def ask(question: str, model: str | None = None) -> tuple[str, str]:
    """Single-turn query helper for backwards compatibility."""
    if not question.strip():
        return "", "Enter a question."
    payload: dict[str, str] = {"question": question}
    if model:
        payload["model"] = model
    try:
        response = httpx.post(f"{API_URL}/query", json=payload, timeout=120)
        response.raise_for_status()
        data = response.json()
        sources = "\n".join(f"- {item['filename']} page {item['page']} (score {item['score']:.3f})" for item in data["sources"])
        return data["answer"], sources or "No supporting sources."
    except (httpx.HTTPError, ValueError) as exc:
        return "", f"Query failed: {exc}"


def chat(
    message: str,
    history: list[list[str]] | None,
    model: str = DEFAULT_CHAT_MODEL,
    selected_doc_id: str | None = None,
) -> tuple[str, list[list[str]], str]:
    """Synchronous conversational query handler maintaining chat history and source citations."""
    current_history = history or []
    query_text = message.strip()
    if not query_text:
        return "", current_history, "Please enter a valid question."

    payload: dict[str, object] = {
        "question": query_text,
        "model": model,
        "history": current_history,
    }
    if selected_doc_id and str(selected_doc_id).strip():
        payload["doc_id"] = str(selected_doc_id).strip()

    try:
        response = httpx.post(f"{API_URL}/query", json=payload, timeout=120)
        response.raise_for_status()
        data = response.json()
        answer = data.get("answer", "No response generated.")
        sources_list = data.get("sources", [])

        if sources_list:
            sources_md = "\n".join(
                f"- **{s['filename']}** — Page {s['page']} (similarity: {s['score']:.3f})"
                for s in sources_list
            )
        else:
            sources_md = "No supporting document evidence found."

        updated_history = current_history + [[query_text, answer]]
        return "", updated_history, sources_md
    except httpx.HTTPStatusError as exc:
        error_msg = f"API Error (HTTP {exc.response.status_code}): {exc.response.text}"
        return "", current_history + [[query_text, error_msg]], error_msg
    except (httpx.HTTPError, ValueError) as exc:
        error_msg = f"Service unavailable: {exc}"
        return "", current_history + [[query_text, error_msg]], error_msg


def chat_stream(
    message: str,
    history: list[list[str]] | None,
    model: str = DEFAULT_CHAT_MODEL,
    selected_doc_id: str | None = None,
    interaction_mode: str = "Mode 1: Direct RAG",
    agent_strategy: str = "hybrid_rerank",
    max_tool_calls: int = 5,
) -> Iterator[tuple[str, list[list[str]], str]]:
    """Token-streaming conversational query handler supporting Direct RAG and Agent Assistant."""
    current_history = history or []
    query_text = message.strip()
    if not query_text:
        yield "", current_history, "Please enter a valid question."
        return

    # Route to Mode 2: Agent Assistant if selected
    if interaction_mode == "Mode 2: Agent Assistant":
        from src.agent.orchestrator import AgentOrchestrator
        from src.agent.agent_config import AgentConfig

        yield "", current_history + [[query_text, "🤖 *Agent Assistant analyzing request & selecting tools...*"]], "🤖 Initializing agent loop..."

        try:
            config = AgentConfig(
                model=model,
                retrieval_strategy=agent_strategy,
                max_iterations=int(max_tool_calls),
                timeout_seconds=60.0,
            )
            orchestrator = AgentOrchestrator(config=config)
            state = orchestrator.run(
                user_query=query_text,
                model=model,
                retrieval_strategy=agent_strategy,
            )

            activity_lines = [f"- {act}" for act in state.activity_log]
            activity_md = "\n".join(activity_lines) if activity_lines else "- Completed execution"

            citations_lines = [f"- **{c}**" for c in state.citations]
            citations_md = "\n".join(citations_lines) if citations_lines else "No direct citations retrieved."

            sources_panel = (
                f"### 🤖 Assistant Activity (Agent Mode)\n{activity_md}\n\n"
                f"### 📑 Supporting Citations\n{citations_md}\n\n"
                f"*Execution Latency: {state.total_latency_ms:.1f}ms • Steps: {state.step_count}*"
            )

            final_history = current_history + [[query_text, state.final_answer]]
            yield "", final_history, sources_panel
            return
        except Exception as exc:
            err_msg = f"Agent Assistant execution error: {exc}"
            yield "", current_history + [[query_text, err_msg]], err_msg
            return

    # Mode 1: Legacy Direct RAG via FastAPI Streaming
    payload: dict[str, object] = {
        "question": query_text,
        "model": model,
        "history": current_history,
    }
    if selected_doc_id and str(selected_doc_id).strip():
        payload["doc_id"] = str(selected_doc_id).strip()

    accumulated_answer = ""
    sources_md = "Retrieving relevant context..."
    yield "", current_history + [[query_text, ""]], sources_md

    try:
        with httpx.stream("POST", f"{API_URL}/query/stream", json=payload, timeout=120) as response:
            response.raise_for_status()
            current_event = None
            for line in response.iter_lines():
                if not line:
                    continue
                if line.startswith("event:"):
                    current_event = line.replace("event:", "").strip()
                    continue
                if line.startswith("data:"):
                    raw_data = line.replace("data:", "", 1).strip()
                    try:
                        data = json.loads(raw_data)
                    except json.JSONDecodeError:
                        continue

                    if current_event == "sources":
                        sources_list = data.get("sources", [])
                        if sources_list:
                            sources_md = "\n".join(
                                f"- **{s['filename']}** — Page {s['page']} (similarity: {s['score']:.3f})"
                                for s in sources_list
                            )
                        else:
                            sources_md = "No supporting document evidence found."
                        yield "", current_history + [[query_text, accumulated_answer]], sources_md

                    elif current_event == "token":
                        accumulated_answer += data.get("token", "")
                        yield "", current_history + [[query_text, accumulated_answer]], sources_md

                    elif current_event == "error":
                        err_text = data.get("error", "An error occurred.")
                        yield "", current_history + [[query_text, f"Error: {err_text}"]], err_text
                        return

                    elif current_event == "done":
                        break

            final_history = current_history + [[query_text, accumulated_answer or "No response generated."]]
            yield "", final_history, sources_md
    except httpx.HTTPStatusError as exc:
        err_msg = f"API Error (HTTP {exc.response.status_code})"
        yield "", current_history + [[query_text, err_msg]], err_msg
    except (httpx.HTTPError, ValueError) as exc:
        err_msg = f"Service unavailable: {exc}"
        yield "", current_history + [[query_text, err_msg]], err_msg


def clear_chat() -> tuple[str, list[list[str]], str]:
    """Clear chat conversation and sources display."""
    return "", [], ""


def delete_document(document_id: str | None, confirmed: bool) -> tuple[str, dict[str, object]]:
    if not document_id:
        status, update = refresh_documents()
        return f"Select a document first. {status}", update
    if not confirmed:
        status, update = refresh_documents()
        return f"Check the confirmation box before deleting. {status}", update
    try:
        response = httpx.delete(f"{API_URL}/documents/{document_id}", timeout=30)
        response.raise_for_status()
        status, update = refresh_documents()
        return f"Document deleted. {status}", update
    except httpx.HTTPStatusError as exc:
        status, update = refresh_documents()
        return f"Delete failed (HTTP {exc.response.status_code}). {status}", update
    except httpx.HTTPError as exc:
        status, update = refresh_documents()
        return f"Delete failed: {exc}. {status}", update


def delete_ui(document_id: str | None, confirmed: bool) -> tuple[str, dict[str, object], dict[str, object]]:
    status, doc_update = delete_document(document_id, confirmed)
    choices, _ = _document_choices()
    scope_update = gr.update(choices=_scope_choices(choices), value="")
    return status, doc_update, scope_update


CUSTOM_CSS = """
/* Splash Overlay Screen */
.splash-overlay {
    position: fixed;
    top: 0;
    left: 0;
    width: 100vw;
    height: 100vh;
    background: radial-gradient(circle at center, #111827 0%, #030712 100%);
    z-index: 999999;
    display: flex;
    justify-content: center;
    align-items: center;
    animation: splashFade 2.2s cubic-bezier(0.77, 0, 0.175, 1) forwards;
    pointer-events: none;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
}

@keyframes splashFade {
    0% { opacity: 1; visibility: visible; }
    72% { opacity: 1; visibility: visible; }
    94% { opacity: 0; visibility: visible; }
    100% { opacity: 0; visibility: hidden; display: none; }
}

.splash-content {
    text-align: center;
    display: flex;
    flex-direction: column;
    align-items: center;
    animation: splashContentRise 1.2s ease-out forwards;
}

@keyframes splashContentRise {
    0% { transform: translateY(20px) scale(0.96); opacity: 0; }
    40% { transform: translateY(0) scale(1); opacity: 1; }
    100% { transform: translateY(0) scale(1); opacity: 1; }
}

.splash-spinner-ring {
    position: relative;
    width: 76px;
    height: 76px;
    margin-bottom: 22px;
    border-radius: 50%;
    border: 3px solid rgba(99, 102, 241, 0.15);
    border-top: 3px solid #6366f1;
    border-right: 3px solid #a855f7;
    animation: spin 1s linear infinite;
    box-shadow: 0 0 25px rgba(99, 102, 241, 0.4);
}

.spinner-core {
    position: absolute;
    top: 50%;
    left: 50%;
    width: 14px;
    height: 14px;
    background: #ec4899;
    border-radius: 50%;
    transform: translate(-50%, -50%);
    box-shadow: 0 0 15px #ec4899;
    animation: pulseCore 1.5s ease-in-out infinite alternate;
}

@keyframes spin {
    100% { transform: rotate(360deg); }
}

@keyframes pulseCore {
    0% { transform: translate(-50%, -50%) scale(0.8); opacity: 0.7; }
    100% { transform: translate(-50%, -50%) scale(1.3); opacity: 1; }
}

.splash-title {
    font-size: 40px;
    font-weight: 800;
    letter-spacing: -1px;
    color: #f9fafb;
    margin: 0 0 6px 0;
}

.splash-title span {
    background: linear-gradient(135deg, #6366f1, #a855f7, #ec4899);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.splash-subtitle {
    font-size: 14px;
    color: #9ca3af;
    margin: 0 0 18px 0;
    letter-spacing: 0.3px;
}

.splash-status-bar {
    width: 220px;
    height: 4px;
    background: rgba(255, 255, 255, 0.08);
    border-radius: 9999px;
    overflow: hidden;
    margin-bottom: 20px;
}

.splash-progress-fill {
    height: 100%;
    width: 0%;
    background: linear-gradient(90deg, #6366f1, #a855f7, #06b6d4);
    border-radius: 9999px;
    animation: fillProgress 1.6s cubic-bezier(0.4, 0, 0.2, 1) forwards;
}

@keyframes fillProgress {
    0% { width: 0%; }
    40% { width: 60%; }
    100% { width: 100%; }
}

.splash-features {
    display: flex;
    gap: 10px;
}

.splash-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 5px 12px;
    background: rgba(255, 255, 255, 0.04);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 9999px;
    font-size: 12px;
    color: #d1d5db;
}

.dot-green {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: #10b981;
    box-shadow: 0 0 8px #10b981;
}

.dot-blue {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: #3b82f6;
    box-shadow: 0 0 8px #3b82f6;
}

.dot-purple {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: #a855f7;
    box-shadow: 0 0 8px #a855f7;
}

/* Hero Header on Page */
.hero-banner {
    background: linear-gradient(135deg, rgba(30, 41, 59, 0.75) 0%, rgba(15, 23, 42, 0.9) 100%);
    border: 1px solid rgba(99, 102, 241, 0.25);
    border-radius: 14px;
    padding: 20px 26px;
    margin-bottom: 16px;
    box-shadow: 0 8px 32px -4px rgba(0, 0, 0, 0.35);
    backdrop-filter: blur(12px);
    transition: border-color 0.3s ease;
}

.hero-banner:hover {
    border-color: rgba(99, 102, 241, 0.45);
}

.hero-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    gap: 14px;
    margin-bottom: 14px;
}

.hero-title {
    font-size: 26px;
    font-weight: 800;
    letter-spacing: -0.5px;
    margin: 0;
    color: #f8fafc;
    display: flex;
    align-items: center;
    gap: 10px;
}

.hero-title .highlight {
    background: linear-gradient(135deg, #6366f1, #a855f7);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.hero-version-tag {
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.5px;
    padding: 3px 8px;
    border-radius: 6px;
    background: rgba(99, 102, 241, 0.2);
    color: #818cf8;
    border: 1px solid rgba(99, 102, 241, 0.3);
}

.hero-desc {
    font-size: 14px;
    color: #94a3b8;
    margin: 4px 0 0 0;
}

.system-status-pill {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 6px 14px;
    border-radius: 9999px;
    background: rgba(16, 185, 129, 0.1);
    border: 1px solid rgba(16, 185, 129, 0.3);
    color: #34d399;
    font-size: 12px;
    font-weight: 600;
    letter-spacing: 0.5px;
}

.live-pulse {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: #10b981;
    box-shadow: 0 0 10px #10b981;
    animation: livePulseAnim 1.8s infinite ease-in-out;
}

@keyframes livePulseAnim {
    0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7); }
    70% { transform: scale(1.1); box-shadow: 0 0 0 8px rgba(16, 185, 129, 0); }
    100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0); }
}

.hero-tags {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
}

.tech-chip {
    display: inline-flex;
    align-items: center;
    padding: 4px 10px;
    font-size: 12px;
    font-weight: 500;
    border-radius: 8px;
    background: rgba(255, 255, 255, 0.04);
    color: #cbd5e1;
    border: 1px solid rgba(255, 255, 255, 0.08);
}
"""

WELCOME_HTML = """
<div id="welcome-splash" class="splash-overlay">
    <div class="splash-content">
        <div class="splash-spinner-ring">
            <div class="spinner-core"></div>
        </div>
        <h1 class="splash-title">Local<span>RAG</span></h1>
        <p class="splash-subtitle">Offline AI & Document Intelligence Engine</p>
        <div class="splash-status-bar">
            <div class="splash-progress-fill"></div>
        </div>
        <div class="splash-features">
            <span class="splash-badge"><span class="dot-green"></span> NVIDIA GPU Accelerated</span>
            <span class="splash-badge"><span class="dot-blue"></span> 100% Offline & Private</span>
            <span class="splash-badge"><span class="dot-purple"></span> Hybrid Search + Reranker</span>
        </div>
    </div>
</div>

<div class="hero-banner">
    <div class="hero-header">
        <div class="hero-title-group">
            <h1 class="hero-title">Local<span class="highlight">RAG</span> <span class="hero-version-tag">PRO</span></h1>
            <p class="hero-desc">Ask questions about your local PDF collection with verified offline inference and zero data leakage.</p>
        </div>
        <div class="hero-indicators">
            <div class="system-status-pill">
                <span class="live-pulse"></span>
                <span>SYSTEM ONLINE • 100% OFFLINE</span>
            </div>
        </div>
    </div>
    <div class="hero-tags">
        <span class="tech-chip">⚡ NVIDIA RTX 4050 GPU</span>
        <span class="tech-chip">🔒 Zero Cloud Telemetry</span>
        <span class="tech-chip">🔍 Dense (nomic-embed) + BM25 Hybrid</span>
        <span class="tech-chip">🎯 Cross-Encoder Reranker</span>
        <span class="tech-chip">📑 Page-Level Grounding</span>
    </div>
</div>
"""


with gr.Blocks(title="LocalRAG", css=CUSTOM_CSS) as demo:
    gr.HTML(WELCOME_HTML)

    with gr.Accordion("Live System Resource Monitor", open=True):
        with gr.Row():
            cpu_metric = gr.Textbox(label="CPU Utilization", value="--", interactive=False, scale=1)
            ram_metric = gr.Textbox(label="System RAM (Used / Total)", value="--", interactive=False, scale=2)
            gpu_metric = gr.Textbox(label="GPU Utilization", value="--", interactive=False, scale=2)
            vram_metric = gr.Textbox(label="GPU VRAM (Used / Total)", value="--", interactive=False, scale=2)
            temp_metric = gr.Textbox(label="GPU Temperature", value="--", interactive=False, scale=1)

    with gr.Row():
        with gr.Column(scale=4):
            gr.Markdown("### Document Management")
            upload = gr.File(label="PDF document", file_types=[".pdf"], type="filepath")
            upload_status = gr.Textbox(label="Ingestion status", interactive=False)
            initial_docs = list_documents()
            document_list = gr.Dropdown(label="Indexed documents", choices=initial_docs, type="value", value=None)

            refresh = gr.Button("Refresh documents")
            refresh_status = gr.Textbox(label="Document refresh status", interactive=False)

            confirm_delete = gr.Checkbox(label="I confirm deleting the selected document")
            delete_status = gr.Textbox(label="Deletion status", interactive=False)
            delete = gr.Button("Delete selected document")

        with gr.Column(scale=6):
            gr.Markdown("### Chat & Agent Assistant")
            with gr.Row():
                mode_selector = gr.Radio(
                    label="Interaction Mode",
                    choices=["Mode 1: Direct RAG", "Mode 2: Agent Assistant"],
                    value="Mode 1: Direct RAG",
                    scale=2,
                )
                model_selector = gr.Dropdown(
                    label="Chat Model",
                    choices=AVAILABLE_MODELS,
                    value=DEFAULT_CHAT_MODEL,
                    interactive=True,
                    scale=1,
                )
                search_scope = gr.Dropdown(
                    label="Search Scope",
                    choices=_scope_choices(initial_docs),
                    value="",
                    type="value",
                    scale=2,
                )

            with gr.Accordion("Agent Assistant Configuration", open=False):
                with gr.Row():
                    agent_strategy = gr.Dropdown(
                        label="Retrieval Strategy",
                        choices=["hybrid_rerank", "bm25_rerank", "hybrid", "bm25", "dense"],
                        value="hybrid_rerank",
                        scale=2,
                    )
                    max_tool_calls = gr.Slider(
                        label="Max Agent Tool Calls",
                        minimum=1,
                        maximum=5,
                        value=5,
                        step=1,
                        scale=1,
                    )

            chatbot = gr.Chatbot(label="Conversation History", type="tuples", height=350)
            with gr.Row():
                question = gr.Textbox(label="Question", placeholder="Ask a question about the indexed documents...", scale=4)
                ask_btn = gr.Button("Ask", variant="primary", scale=1)

            clear_btn = gr.Button("Clear Conversation")
            sources = gr.Markdown(label="Supporting Sources & Activity")

            # Document management UI interactions
            upload.change(upload_ui, inputs=upload, outputs=[upload_status, document_list, search_scope])
            refresh.click(refresh_ui, outputs=[refresh_status, document_list, search_scope])
            delete.click(delete_ui, inputs=[document_list, confirm_delete], outputs=[delete_status, document_list, search_scope])

            # Streaming chat interaction with Mode 1 / Mode 2 routing
            chat_inputs = [question, chatbot, model_selector, search_scope, mode_selector, agent_strategy, max_tool_calls]
            ask_btn.click(chat_stream, inputs=chat_inputs, outputs=[question, chatbot, sources])
            question.submit(chat_stream, inputs=chat_inputs, outputs=[question, chatbot, sources])
            clear_btn.click(clear_chat, outputs=[question, chatbot, sources])

    # Live system resource monitoring updates
    monitor_timer = gr.Timer(value=2.0)
    monitor_timer.tick(
        fn=format_metrics_for_display,
        inputs=[],
        outputs=[cpu_metric, ram_metric, gpu_metric, vram_metric, temp_metric],
    )
    demo.load(
        fn=format_metrics_for_display,
        inputs=[],
        outputs=[cpu_metric, ram_metric, gpu_metric, vram_metric, temp_metric],
    )


if __name__ == "__main__":
    host = os.getenv("FRONTEND_HOST", "127.0.0.1")
    port = int(os.getenv("FRONTEND_PORT", "7860"))
    demo.launch(server_name=host, server_port=port)
