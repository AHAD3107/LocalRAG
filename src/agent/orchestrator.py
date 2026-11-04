"""Autonomous agent orchestrator for LocalRAG."""
from __future__ import annotations

import json
import logging
import re
import time
import urllib.request
from typing import Any

from config import settings
from src.agent.agent_config import AgentConfig
from src.agent.agent_state import AgentState
from src.agent.conversation_manager import ConversationManager, default_conversation_manager
from src.agent.exceptions import AgentTimeoutError, MaxIterationsExceededError
from src.agent.tool_executor import ToolExecutor
from src.agent.tool_registry import ToolRegistry, default_tool_registry

logger = logging.getLogger(__name__)


class AgentOrchestrator:
    """Coordinates reasoning, tool selection, execution, and grounded answer synthesis."""

    def __init__(
        self,
        config: AgentConfig | None = None,
        registry: ToolRegistry | None = None,
        conversation_manager: ConversationManager | None = None,
    ) -> None:
        self.config = config or AgentConfig()
        self.registry = registry or default_tool_registry
        self.executor = ToolExecutor(self.registry)
        self.conversation_manager = conversation_manager or default_conversation_manager

    def _call_ollama(
        self,
        messages: list[dict[str, Any]],
        model: str,
        timeout: float,
    ) -> dict[str, Any]:
        """Execute a single chat inference call to local Ollama."""
        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": 0.1},
        }

        # Attempt to pass tools schema to Ollama if model allows
        # If model rejects tools with 400, fallback cleanly to prompt-only
        try_with_tools = self.registry.get_ollama_tools_schema()
        payload_with_tools = dict(payload, tools=try_with_tools)

        req_data = json.dumps(payload_with_tools).encode("utf-8")
        req = urllib.request.Request(
            f"{settings.ollama_base_url}/api/chat",
            data=req_data,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as err:
            err_body = err.read().decode("utf-8", errors="replace")
            # If model does not support tools parameter (e.g. llama3:8b), retry without tools parameter
            if err.code == 400 and "does not support tools" in err_body:
                logger.info("Model '%s' does not support native tools parameter; using prompt-based action dispatcher.", model)
                req_data_plain = json.dumps(payload).encode("utf-8")
                req_plain = urllib.request.Request(
                    f"{settings.ollama_base_url}/api/chat",
                    data=req_data_plain,
                    headers={"Content-Type": "application/json"},
                )
                with urllib.request.urlopen(req_plain, timeout=timeout) as resp2:
                    return json.loads(resp2.read().decode("utf-8"))
            raise RuntimeError(f"Ollama HTTP error {err.code}: {err_body}") from err

    def _extract_tool_calls(self, response_message: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
        """Extract tool calls from either Ollama tool_calls or structured JSON in content."""
        calls = []

        # 1. Native tool_calls
        raw_tool_calls = response_message.get("tool_calls", [])
        if raw_tool_calls:
            for tc in raw_tool_calls:
                fn = tc.get("function", {})
                name = fn.get("name")
                args = fn.get("arguments", {})
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except Exception:
                        args = {}
                if name:
                    calls.append((name, args if isinstance(args, dict) else {}))
            return calls

        # 2. Extract JSON action blocks in message content
        content = response_message.get("content", "")
        # Look for markdown code blocks containing json or action object
        patterns = [
            r"```(?:json)?\s*(\{[\s\S]*?\})\s*```",
            r'(\{\s*"(?:action|name|tool)"\s*:\s*"[^"]+[\s\S]*?\})',
        ]
        for pat in patterns:
            matches = re.findall(pat, content)
            for match in matches:
                try:
                    parsed = json.loads(match)
                    if isinstance(parsed, dict):
                        # Extract tool name and arguments
                        name = parsed.get("action") or parsed.get("name") or parsed.get("tool")
                        args = parsed.get("parameters") or parsed.get("arguments") or parsed.get("params") or {}
                        # If the parameters are flattened into the dict
                        if not args and isinstance(name, str):
                            args = {k: v for k, v in parsed.items() if k not in ("action", "name", "tool")}
                        if name and isinstance(name, str) and self.registry.has_tool(name):
                            calls.append((name, args if isinstance(args, dict) else {}))
                except Exception:
                    continue

        return calls

    def _build_system_prompt(self, document_scope: str | None = None) -> str:
        tools_doc = self.registry.get_prompt_tools_documentation()
        
        doc_listing = ""
        try:
            from src.tools.document_tools import list_documents
            docs = list_documents()
            if docs:
                lines = [f"- Filename: '{d['filename']}' (ID: '{d['document_id']}', Pages: {d.get('page_count', 1)})" for d in docs]
                doc_listing = "AVAILABLE INDEXED DOCUMENTS IN LOCALRAG:\n" + "\n".join(lines) + "\n"
                if document_scope:
                    doc_listing += f"USER TARGET / SELECTED SEARCH SCOPE: '{document_scope}'\n"
        except Exception:
            pass

        return (
            "You are LocalRAG Intelligent Assistant, a privacy-first technical assistant operating 100% offline.\n"
            "You have access to local documents and specialized tools. When the user asks a question, determine "
            "whether you need to retrieve or analyze documents. If you need information, call one of the available tools.\n\n"
            f"{doc_listing}\n"
            "AVAILABLE TOOLS:\n"
            f"{tools_doc}\n\n"
            "INSTRUCTIONS FOR TOOL INVOCATION:\n"
            "1. If you need to call a tool, respond with a single JSON block:\n"
            "```json\n"
            '{"action": "<tool_name>", "parameters": {<arguments>}}\n'
            "```\n"
            "2. When invoking tools that require document_id or filename (e.g. summarize_document, get_document_info, retrieve_passages), "
            "ALWAYS supply one of the exact document filenames or IDs from the available indexed documents above (e.g. 'machine_learning_algorithms.pdf'). "
            "Never use generic placeholder strings like 'document_id'. If the user refers to 'the document' or 'uploaded document', use the target document or first document from the list.\n"
            "3. If evidence is insufficient or a question asks about topics not in the documents, state that clearly without guessing.\n"
            "4. When you have sufficient information to answer the user's question, provide a comprehensive, direct answer "
            "with exact document and page citations.\n"
        )

    def run(
        self,
        user_query: str,
        session_id: str | None = None,
        model: str | None = None,
        retrieval_strategy: str | None = None,
        document_scope: str | None = None,
    ) -> AgentState:
        """Execute the agent loop for the given query."""
        start_time = time.perf_counter()
        target_model = model or self.config.model
        clean_query = user_query.strip()
        session = self.conversation_manager.get_or_create_session(session_id)

        state = AgentState(
            session_id=session.session_id,
            user_query=clean_query,
        )

        state.add_activity("Analyzing user request and identifying required information")

        # Build initial message stack with document scope awareness
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self._build_system_prompt(document_scope)},
        ]
        # Include session turns (previous history)
        for turn in session.turns[-self.config.max_history_turns:]:
            messages.append({"role": turn.role, "content": turn.content})

        # Append current user prompt
        messages.append({"role": "user", "content": clean_query})

        last_tool_call_signature = None

        for iteration in range(self.config.max_iterations):
            state.step_count = iteration + 1
            elapsed_total = time.perf_counter() - start_time
            if elapsed_total > self.config.timeout_seconds:
                state.add_activity(f"Execution reached timeout ({self.config.timeout_seconds}s)")
                break

            time_left = max(5.0, self.config.timeout_seconds - elapsed_total)
            try:
                response = self._call_ollama(messages, target_model, timeout=time_left)
            except Exception as exc:
                logger.error("Error communicating with Ollama in iteration %d: %s", iteration, exc)
                state.add_activity(f"Ollama inference error: {exc}")
                break

            resp_msg = response.get("message", {})
            content = resp_msg.get("content", "").strip()
            tool_calls = self._extract_tool_calls(resp_msg)

            # If no tools called, this is the final answer
            if not tool_calls:
                state.final_answer = content
                state.add_activity("Synthesizing final grounded response")
                break

            # Execute tool calls
            executed_any = False
            for tool_name, args in tool_calls:
                # Anti-cycle detection: prevent identical consecutive calls
                call_sig = f"{tool_name}:{json.dumps(args, sort_keys=True)}"
                if call_sig == last_tool_call_signature:
                    logger.warning("Detected duplicate consecutive tool call: %s. Terminating loop.", call_sig)
                    state.add_activity("Detected redundant tool call; formulating final response")
                    state.final_answer = ""
                    break
                last_tool_call_signature = call_sig

                # User-facing activity message
                if tool_name == "list_documents":
                    state.add_activity("Listing available indexed documents")
                elif tool_name == "get_document_info":
                    state.add_activity(f"Retrieving metadata for document '{args.get('document_id')}'")
                elif tool_name == "search_documents":
                    state.add_activity(f"Searching documents for: '{args.get('query', '')[:40]}'")
                elif tool_name == "retrieve_passages":
                    state.add_activity(f"Retrieving page {args.get('page_number')} from '{args.get('document_id')}'")
                elif tool_name == "summarize_document":
                    state.add_activity(f"Generating summary for document '{args.get('document_id')}'")
                elif tool_name == "compare_documents":
                    state.add_activity(f"Comparing documents {args.get('document_ids')}")
                elif tool_name == "generate_study_questions":
                    state.add_activity("Generating curriculum study questions")
                else:
                    state.add_activity(f"Executing tool '{tool_name}'")

                # Pass retrieval strategy override if search tool
                if tool_name == "search_documents" and retrieval_strategy and "retrieval_strategy" not in args:
                    args["retrieval_strategy"] = retrieval_strategy

                result, record = self.executor.execute(tool_name, args)
                state.records.append(record)
                executed_any = True

                # Extract citations from tool result
                if isinstance(result, dict):
                    if "citation" in result and result["citation"]:
                        state.add_citation(result["citation"])
                    if "citations" in result and isinstance(result["citations"], list):
                        for c in result["citations"]:
                            state.add_citation(c)
                    if "results" in result and isinstance(result["results"], list):
                        for item in result["results"]:
                            if "citation" in item:
                                state.add_citation(item["citation"])

                # High-level synthesis tools already return completed user-facing answers
                if tool_name in ("summarize_document", "compare_documents", "generate_study_questions"):
                    if isinstance(result, dict) and not result.get("error"):
                        if "summary" in result and result["summary"]:
                            topics = result.get("key_topics", [])
                            topics_str = "\n".join([f"- {t}" for t in topics]) if topics else ""
                            if topics_str:
                                state.final_answer = f"{result['summary']}\n\n**Key Topics:**\n{topics_str}"
                            else:
                                state.final_answer = result["summary"]
                            state.add_activity("Synthesizing final grounded response")
                            break
                        elif "comparison" in result and result["comparison"]:
                            state.final_answer = result["comparison"]
                            state.add_activity("Synthesizing final grounded response")
                            break
                        elif "questions_content" in result and result["questions_content"]:
                            state.final_answer = result["questions_content"]
                            state.add_activity("Synthesizing final grounded response")
                            break

                # Feed observation back to model
                messages.append({
                    "role": "assistant",
                    "content": f'{{"action": "{tool_name}", "parameters": {json.dumps(args)}}}',
                })
                # Truncate large observation payloads to protect context window
                obs_str = json.dumps(result)
                if len(obs_str) > 3000:
                    obs_str = obs_str[:3000] + "... [Observation truncated to prevent context overflow]"

                messages.append({
                    "role": "user",
                    "content": f"[Tool Observation from {tool_name}]:\n{obs_str}\n\nNow use this information to answer the user request or determine the next step.",
                })

            if state.final_answer:
                break

            if not executed_any:
                if not state.final_answer:
                    state.final_answer = content or "Completed requested actions."
                break

        # Check if final_answer is empty, an echoed tool call, a raw JSON tool call, or if anti-cycle terminated
        is_raw_tool_call = False
        if state.final_answer:
            stripped = state.final_answer.strip()
            if (
                stripped.startswith("Invoking tool:")
                or "Invoking tool:" in stripped
                or (('{"action":' in stripped or '"action":' in stripped) and len(stripped) < 300)
                or (('{"name":' in stripped or '"tool":' in stripped) and len(stripped) < 300)
            ):
                is_raw_tool_call = True

        # Fallback or extract synthesized result if final answer was raw tool JSON or empty
        if not state.final_answer or is_raw_tool_call:
            if state.records:
                last_rec = state.records[-1]
                if isinstance(last_rec.result, dict) and "summary" in last_rec.result:
                    topics = last_rec.result.get("key_topics", [])
                    topics_str = "\n".join([f"- {t}" for t in topics]) if topics else ""
                    if topics_str:
                        state.final_answer = f"{last_rec.result['summary']}\n\n**Key Topics:**\n{topics_str}"
                    else:
                        state.final_answer = last_rec.result["summary"]
                elif isinstance(last_rec.result, dict) and "comparison" in last_rec.result:
                    state.final_answer = last_rec.result["comparison"]
                elif isinstance(last_rec.result, dict) and "questions_content" in last_rec.result:
                    state.final_answer = last_rec.result["questions_content"]
                elif isinstance(last_rec.result, dict) and "answer" in last_rec.result:
                    state.final_answer = last_rec.result["answer"]
                elif isinstance(last_rec.result, dict) and "results" in last_rec.result:
                    passages = []
                    for r in last_rec.result["results"]:
                        c = r.get("citation", "")
                        t = r.get("text", "")
                        passages.append(f"**[{c}]**\n{t}")
                    state.final_answer = "\n\n".join(passages)
                else:
                    state.final_answer = "Completed tool execution. " + json.dumps(last_rec.result, indent=2)
            else:
                state.final_answer = "I processed your request, but was unable to retrieve supporting evidence from the local documents."

        state.abstained = "cannot find information" in state.final_answer.lower()
        state.total_latency_ms = round((time.perf_counter() - start_time) * 1000, 2)

        # Update conversation session
        self.conversation_manager.add_user_message(state.session_id, clean_query)
        self.conversation_manager.add_assistant_message(
            state.session_id,
            state.final_answer,
            activity=state.activity_log,
            citations=state.citations,
        )

        return state

