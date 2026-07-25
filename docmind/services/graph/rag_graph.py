"""The RAG workflow as an official LangGraph ``StateGraph``.

Pipeline::

    START
      → load_conversation
      → [query_expansion]      (optional)
      → retrieve
      → [rerank]               (optional)
      → generate
      → save
      → END

Every node is a small pure-ish function that reads/writes :class:`RAGState`.
Orchestration, edges and compilation are entirely LangGraph's.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Callable

from docmind.config.logging import get_logger
from docmind.config.settings import settings
from docmind.state import RAGState

if TYPE_CHECKING:
    from langchain_core.documents import Document
    from langchain_core.language_models.chat_models import BaseChatModel
    from langchain_core.prompts import ChatPromptTemplate

    from docmind.services.memory import ChatMemoryService
    from docmind.services.persistence import MetadataStore
    from docmind.services.retrieval import RetrievalService

logger = get_logger(__name__)


def _text(content) -> str:
    if isinstance(content, list):
        return "".join(
            part.get("text", "") if isinstance(part, dict) else str(part) for part in content
        )
    return str(content or "")


def _fallback_title(question: str) -> str:
    words = (question or "").strip().split()
    return " ".join(words[:5]) or "New Chat"


def format_context(documents: list["Document"]) -> tuple[str, list[dict]]:
    """Build the LLM context string and de-duplicated source list."""
    context_parts: list[str] = []
    sources: list[dict] = []
    seen: set[str] = set()
    for index, doc in enumerate(documents, start=1):
        meta = doc.metadata or {}
        filename = meta.get("filename", "source")
        citation = meta.get("citation") or filename
        context_parts.append(f"[Source {index}: {citation}]\n{doc.page_content}")

        key = meta.get("document_id") or filename
        if key in seen:
            continue
        seen.add(key)
        link = ""
        try:
            candidate = Path(settings.UPLOAD_DIR) / filename
            if candidate.exists():
                link = candidate.resolve().as_uri()
        except (ValueError, OSError):
            link = ""
        sources.append(
            {
                "document_id": meta.get("document_id", ""),
                "name": filename,
                "link": link,
                "citation": citation,
            }
        )
    return "\n\n".join(context_parts), sources


@dataclass
class GraphContext:
    """Dependencies injected into the graph nodes."""

    get_chat_model: Callable[[], "BaseChatModel"]
    retrieval: "RetrievalService"
    memory: "ChatMemoryService"
    metadata: "MetadataStore"
    rag_prompt: "ChatPromptTemplate"
    general_prompt: "ChatPromptTemplate"
    title_prompt: "ChatPromptTemplate"
    expansion_prompt: "ChatPromptTemplate"
    enable_query_expansion: bool = False
    enable_rerank: bool = False


def build_rag_graph(ctx: GraphContext):
    """Compile the RAG StateGraph for the given dependencies."""
    from langgraph.graph import END, START, StateGraph

    def load_conversation(state: RAGState) -> RAGState:
        history = ctx.memory.recent_messages(state["session_id"])
        return {"chat_history": history}

    def query_expansion(state: RAGState) -> RAGState:
        question = state["question"]
        try:
            chain = ctx.expansion_prompt | ctx.get_chat_model()
            response = chain.invoke({"question": question})
            extra = [line.strip() for line in _text(response.content).splitlines() if line.strip()]
        except Exception as exc:  # expansion is best-effort
            logger.warning("Query expansion failed: %s", exc)
            extra = []
        return {"queries": [question, *extra[:3]]}

    def retrieve(state: RAGState) -> RAGState:
        if not state.get("rag_mode", True):
            return {"documents": [], "context": "", "sources": []}
        queries = state.get("queries") or [state["question"]]
        documents = ctx.retrieval.retrieve(queries)
        context, sources = format_context(documents)
        return {"documents": documents, "context": context, "sources": sources}

    def rerank(state: RAGState) -> RAGState:
        documents = ctx.retrieval.rerank_by_score(state["question"], state.get("documents", []))
        context, sources = format_context(documents)
        return {"documents": documents, "context": context, "sources": sources}

    def generate(state: RAGState) -> RAGState:
        rag_mode = state.get("rag_mode", True)
        history = state.get("chat_history", [])
        model = ctx.get_chat_model()
        if rag_mode:
            chain = ctx.rag_prompt | model
            variables = {
                "question": state["question"],
                "context": state.get("context", "") or "No context was found.",
                "chat_history": history,
            }
        else:
            chain = ctx.general_prompt | model
            variables = {"question": state["question"], "chat_history": history}

        response = chain.invoke(variables)
        answer = _text(response.content)
        usage = getattr(response, "usage_metadata", None) or {}
        debug = {
            "mode": "rag" if rag_mode else "general_chat",
            "orchestration": "LangGraph",
            "llm_stack": "LangChain",
            "retrieved_chunks": len(state.get("documents", [])),
            "unique_sources": len(state.get("sources", [])),
            "token_usage": usage,
        }
        return {"answer": answer, "debug": debug}

    def save(state: RAGState) -> RAGState:
        session_id = state["session_id"]
        ctx.memory.add_exchange(
            session_id,
            state["question"],
            state.get("answer", ""),
            state.get("sources", []),
        )
        current_title = ctx.metadata.session_title(session_id)
        if current_title in (None, "", "New Chat"):
            try:
                chain = ctx.title_prompt | ctx.get_chat_model()
                raw = _text(chain.invoke({"question": state["question"]}).content)
                words = raw.strip().strip('"').split()
                title = " ".join(words[:5]) or _fallback_title(state["question"])
            except Exception as exc:
                logger.warning("Title generation failed: %s", exc)
                title = _fallback_title(state["question"])
            ctx.metadata.update_session(session_id, title=title)
        else:
            ctx.metadata.touch_session(session_id)
        return {}

    graph = StateGraph(RAGState)
    graph.add_node("load_conversation", load_conversation)
    graph.add_node("retrieve", retrieve)
    graph.add_node("generate", generate)
    graph.add_node("save", save)

    graph.add_edge(START, "load_conversation")

    if ctx.enable_query_expansion:
        graph.add_node("query_expansion", query_expansion)
        graph.add_edge("load_conversation", "query_expansion")
        graph.add_edge("query_expansion", "retrieve")
    else:
        graph.add_edge("load_conversation", "retrieve")

    if ctx.enable_rerank:
        graph.add_node("rerank", rerank)
        graph.add_edge("retrieve", "rerank")
        graph.add_edge("rerank", "generate")
    else:
        graph.add_edge("retrieve", "generate")

    graph.add_edge("generate", "save")
    graph.add_edge("save", END)

    return graph.compile()
