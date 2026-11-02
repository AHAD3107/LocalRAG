"""Conversational query contextualization for follow-up questions."""
from __future__ import annotations

import re
from typing import Sequence

from src.embeddings.ollama import OllamaClient, OllamaError


def format_chat_history(history: Sequence[Sequence[str] | dict[str, str]], max_turns: int = 3) -> str:
    """Format recent conversation turns into a clean text transcript."""
    lines: list[str] = []
    turns = history[-max_turns:]
    for turn in turns:
        if isinstance(turn, dict):
            user_msg = turn.get("user") or turn.get("question") or turn.get("role_user") or ""
            bot_msg = turn.get("assistant") or turn.get("answer") or turn.get("role_assistant") or ""
        elif isinstance(turn, (list, tuple)) and len(turn) >= 2:
            user_msg = str(turn[0])
            bot_msg = str(turn[1])
        else:
            continue
        if user_msg.strip():
            lines.append(f"User: {user_msg.strip()}")
        if bot_msg.strip():
            lines.append(f"Assistant: {bot_msg.strip()}")
    return "\n".join(lines)


def build_contextualization_prompt(question: str, history_text: str) -> str:
    return (
        "You are a search query reformulation assistant. "
        "Given the chat history below and a follow-up user question, "
        "rewrite the user question into a complete, standalone question that can be understood without the chat history. "
        "Follow these rules strictly:\n"
        "1. If the question contains pronouns or references to earlier topics (e.g., 'it', 'they', 'this project', 'that deadline'), replace them with the explicit nouns from the history.\n"
        "2. If the question is already complete and standalone, return it exactly as is.\n"
        "3. Do NOT answer the question. Do NOT add new facts or guess answers.\n"
        "4. Output ONLY the rewritten question text and nothing else.\n\n"
        f"Chat History:\n{history_text}\n\n"
        f"Follow-up Question: {question}\n\n"
        "Standalone Question:"
    )


def contextualize_query(
    question: str,
    history: Sequence[Sequence[str] | dict[str, str]] | None,
    client: OllamaClient,
    model: str,
) -> str:
    """
    Contextualize a user query using prior chat history if needed.
    Returns the standalone rewritten query, or the original query if history is empty.
    """
    query = question.strip()
    if not history or not query:
        return query

    history_text = format_chat_history(history)
    if not history_text:
        return query

    prompt = build_contextualization_prompt(query, history_text)
    try:
        rewritten = client.chat(prompt, model).strip()
        rewritten = re.sub(r'^(Standalone Question:\s*|["\']+|Q:\s*)', "", rewritten, flags=re.IGNORECASE)
        rewritten = re.sub(r'["\']+$', "", rewritten).strip()
        return rewritten if rewritten else query
    except OllamaError:
        return query

