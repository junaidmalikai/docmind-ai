"""Typed state that flows through the LangGraph RAG pipeline.

Channel value types are intentionally kept as plain builtins (``list``/``dict``)
so LangGraph can resolve the ``TypedDict`` hints without importing heavy
LangChain modules at application-startup time. The concrete element types are
``langchain_core`` ``BaseMessage`` / ``Document`` objects at runtime.
"""

from __future__ import annotations

from typing import Any, TypedDict


class RAGState(TypedDict, total=False):
    """Shared state for the RAG graph (``total=False`` — nodes add keys)."""

    # Inputs
    question: str
    session_id: str
    rag_mode: bool

    # Populated by nodes
    queries: list           # query-expansion output (list[str])
    chat_history: list      # list[BaseMessage] loaded from memory
    documents: list         # list[Document] retrieved from the vector store
    context: str
    sources: list           # list[dict] citation payloads
    answer: str
    debug: dict[str, Any]
