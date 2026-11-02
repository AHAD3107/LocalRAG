"""Gradio frontend that calls the LocalRAG API."""
from __future__ import annotations

import json
import os
from typing import Iterator

import gradio as gr
import httpx


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
        response = httpx.get(f"{API_URL}/documents", timeout=30)
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
        return [], f"Could not load documents: {exc}."


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
        return f"Upload failed (HTTP {exc.response.status_code}). {refresh_status}", update
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
) -> Iterator[tuple[str, list[list[str]], str]]:
    """Token-streaming conversational query handler."""
    current_history = history or []
    query_text = message.strip()
    if not query_text:
        yield "", current_history, "Please enter a valid question."
        return

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


with gr.Blocks(title="LocalRAG") as demo:
    gr.Markdown("# LocalRAG\nAsk questions about your local PDF collection with verified offline inference.")

    with gr.Row():
        with gr.Column(scale=4):
            gr.Markdown("### Document Management")
            upload = gr.File(label="PDF document", file_types=[".pdf"], type="filepath")
            upload_status = gr.Textbox(label="Ingestion status", interactive=False)
            document_list = gr.Dropdown(label="Indexed documents", choices=list_documents(), type="value", value=None)

            refresh = gr.Button("Refresh documents")
            refresh_status = gr.Textbox(label="Document refresh status", interactive=False)

            confirm_delete = gr.Checkbox(label="I confirm deleting the selected document")
            delete_status = gr.Textbox(label="Deletion status", interactive=False)
            delete = gr.Button("Delete selected document")

        with gr.Column(scale=6):
            gr.Markdown("### Chat & Question Answering")
            with gr.Row():
                model_selector = gr.Dropdown(
                    label="Chat Model",
                    choices=AVAILABLE_MODELS,
                    value=DEFAULT_CHAT_MODEL,
                    interactive=True,
                    scale=1,
                )
                search_scope = gr.Dropdown(
                    label="Search Scope",
                    choices=_scope_choices(list_documents()),
                    value="",
                    type="value",
                    scale=2,
                )

            chatbot = gr.Chatbot(label="Conversation History", type="tuples", height=350)
            with gr.Row():
                question = gr.Textbox(label="Question", placeholder="Ask a question about the indexed documents...", scale=4)
                ask_btn = gr.Button("Ask", variant="primary", scale=1)

            clear_btn = gr.Button("Clear Conversation")
            sources = gr.Markdown(label="Supporting Sources")

            # Document management UI interactions
            upload.change(upload_ui, inputs=upload, outputs=[upload_status, document_list, search_scope])
            refresh.click(refresh_ui, outputs=[refresh_status, document_list, search_scope])
            delete.click(delete_ui, inputs=[document_list, confirm_delete], outputs=[delete_status, document_list, search_scope])

            # Streaming chat interaction
            ask_btn.click(chat_stream, inputs=[question, chatbot, model_selector, search_scope], outputs=[question, chatbot, sources])
            question.submit(chat_stream, inputs=[question, chatbot, model_selector, search_scope], outputs=[question, chatbot, sources])
            clear_btn.click(clear_chat, outputs=[question, chatbot, sources])


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.getenv("FRONTEND_PORT", "7860")))
