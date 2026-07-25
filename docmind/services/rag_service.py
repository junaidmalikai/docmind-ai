"""Application-facing RAG orchestrator.

This is a thin dependency-injection facade: it wires official LangChain services
and a compiled LangGraph together and exposes a small, stable API to the
Streamlit UI. It contains no bespoke RAG logic of its own.
"""

from __future__ import annotations

import threading
import time
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Iterator, Optional

from docmind.config.logging import get_logger
from docmind.config.providers import provider_settings, resolve_provider
from docmind.config.settings import Settings, settings as default_settings
from docmind.models import DocumentInfo, LLMSettings, RAGResponse, SourceDocument, TokenUsage
from docmind.services.embeddings import create_embeddings
from docmind.services.graph import GraphContext, build_rag_graph
from docmind.services.llm import LLMConfigurationError, create_chat_model
from docmind.services.loaders import SUPPORTED_EXTENSIONS, DocumentLoaderService
from docmind.services.memory import ChatMemoryService
from docmind.services.persistence import MetadataStore
from docmind.services.prompts import (
    build_general_prompt,
    build_query_expansion_prompt,
    build_rag_prompt,
    build_title_prompt,
)
from docmind.services.retrieval import RetrievalService
from docmind.services.vectorstore import VectorStoreService
from docmind.utils.helpers import ensure_dir, safe_filename, utc_now

if TYPE_CHECKING:
    from langchain_core.language_models.chat_models import BaseChatModel

logger = get_logger(__name__)

__all__ = ["RAGService", "LLMConfigurationError", "SUPPORTED_EXTENSIONS"]


class RAGService:
    """Composes loaders, embeddings, vector store, retrieval, memory and the graph."""

    def __init__(self, config: Optional[Settings] = None, llm_settings: Optional[LLMSettings] = None):
        self.config = config or default_settings
        for folder in (self.config.UPLOAD_DIR, self.config.CHROMA_PATH, self.config.CACHE_DIR):
            ensure_dir(folder)

        self.llm_settings = llm_settings or self._default_llm_settings()
        self._chat_model: Optional["BaseChatModel"] = None
        self._chat_model_lock = threading.Lock()

        self.loader = DocumentLoaderService(self.config.CHUNK_SIZE, self.config.CHUNK_OVERLAP)
        self.vector_store = VectorStoreService(
            persist_directory=self.config.CHROMA_PATH,
            embedding_provider=self.config.EMBEDDING_PROVIDER,
            embedding_model=self.config.EMBEDDING_MODEL,
        )
        self.retrieval = RetrievalService(self.vector_store, self.config.MAX_RESULTS)
        self.metadata = MetadataStore(self.config.sqlite_url)
        self.memory = ChatMemoryService(self.config.sqlite_url, self.config.MAX_HISTORY)

        self._graph = None
        self._warm_up_started = False

    # -- Configuration --------------------------------------------------
    def _default_llm_settings(self) -> LLMSettings:
        provider = resolve_provider(self.config.DEFAULT_PROVIDER)
        defaults = provider_settings(provider)
        return LLMSettings(
            provider=provider,
            model=defaults["model"],
            api_key=defaults["api_key"],
            base_url=defaults["base_url"],
            temperature=self.config.TEMPERATURE,
            top_p=self.config.TOP_P,
            max_tokens=self.config.MAX_TOKENS,
            streaming=self.config.STREAMING,
        )

    def update_llm(self, llm_settings: LLMSettings) -> None:
        self.llm_settings = llm_settings
        with self._chat_model_lock:
            self._chat_model = None

    def update_rag_params(
        self,
        *,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        max_results: Optional[int] = None,
    ) -> None:
        """Apply UI-driven chunking / retrieval knobs at runtime.

        ``chunk_size`` / ``chunk_overlap`` affect the next upload only.
        ``max_results`` (K) applies immediately to the next query.
        """
        if chunk_size is not None:
            size = max(100, int(chunk_size))
            self.loader.chunk_size = size
        if chunk_overlap is not None:
            # Overlap must stay strictly smaller than chunk size.
            overlap = max(0, int(chunk_overlap))
            self.loader.chunk_overlap = min(overlap, max(0, self.loader.chunk_size - 1))
        if max_results is not None:
            self.retrieval.k = max(1, int(max_results))

    def _get_chat_model(self) -> "BaseChatModel":
        if self._chat_model is None:
            with self._chat_model_lock:
                if self._chat_model is None:
                    self._chat_model = create_chat_model(self.llm_settings)
        return self._chat_model

    # -- Graph ----------------------------------------------------------
    def _get_graph(self):
        if self._graph is None:
            ctx = GraphContext(
                get_chat_model=self._get_chat_model,
                retrieval=self.retrieval,
                memory=self.memory,
                metadata=self.metadata,
                rag_prompt=build_rag_prompt(),
                general_prompt=build_general_prompt(),
                title_prompt=build_title_prompt(),
                expansion_prompt=build_query_expansion_prompt(),
                enable_query_expansion=self.config.ENABLE_QUERY_EXPANSION,
                enable_rerank=self.config.ENABLE_RERANK,
            )
            self._graph = build_rag_graph(ctx)
        return self._graph

    def warm_up_async(self) -> None:
        """Preload embeddings + Chroma + the graph off the UI thread, once."""
        if self._warm_up_started:
            return
        self._warm_up_started = True

        def _load() -> None:
            try:
                self._get_graph()
                self.vector_store.warm_up()
            except Exception as exc:
                logger.warning("Warm-up failed: %s", exc)
                self._warm_up_started = False

        threading.Thread(target=_load, name="docmind-warmup", daemon=True).start()

    # -- Ingestion ------------------------------------------------------
    def ingest_bytes(self, filename: str, content: bytes) -> tuple[DocumentInfo, int]:
        safe_name = safe_filename(filename)
        target = Path(self.config.UPLOAD_DIR) / safe_name
        target.write_bytes(content)
        return self.ingest_file(str(target))

    def ingest_file(self, file_path: str) -> tuple[DocumentInfo, int]:
        path = Path(file_path)
        doc_id, chunks = self.loader.load_and_split(str(path))
        # Re-indexing a file replaces its previous chunks.
        self.vector_store.delete_document(doc_id)
        n_chunks = self.vector_store.add_documents(chunks)
        info = DocumentInfo(
            id=doc_id,
            filename=path.name,
            file_type=path.suffix.lstrip("."),
            source_path=str(path),
            chunk_count=n_chunks,
            created_at=utc_now(),
        )
        self.metadata.upsert_document(info)
        return info, n_chunks

    def delete_document(self, document_id: str) -> None:
        self.vector_store.delete_document(document_id)
        self.metadata.delete_document(document_id)

    def list_documents(self) -> list[DocumentInfo]:
        return self.metadata.list_documents()

    def get_document_analytics(self) -> dict:
        stats = self.metadata.analytics()
        return {
            **stats,
            "documents": self.list_documents(),
            "vector_db": "ChromaDB",
            "orchestration": "LangGraph",
            "llm_stack": "LangChain",
            "embedding_model": self.config.EMBEDDING_MODEL,
            "llm_provider": self.llm_settings.provider,
            "llm_model": self.llm_settings.model,
        }

    # -- Conversations --------------------------------------------------
    def create_chat(self) -> str:
        session_id = f"session_{uuid.uuid4().hex}"
        self.metadata.create_session(
            session_id,
            provider=self.llm_settings.provider,
            model=self.llm_settings.model,
        )
        return session_id

    def list_chat_sessions(self, search: str = ""):
        return self.metadata.list_sessions(search=search)

    def load_chat_messages(self, session_id: str):
        return self.memory.load_chat_messages(session_id)

    def rename_chat(self, session_id: str, title: str) -> None:
        self.metadata.update_session(session_id, title=(title or "").strip() or "New Chat")

    def delete_chat(self, session_id: str) -> None:
        self.memory.clear(session_id)
        self.metadata.delete_session(session_id)

    # -- Querying -------------------------------------------------------
    def _ensure_session(self, session_id: Optional[str]) -> str:
        if session_id:
            if self.metadata.session_title(session_id) is None:
                self.metadata.create_session(
                    session_id,
                    provider=self.llm_settings.provider,
                    model=self.llm_settings.model,
                )
            return session_id
        return self.create_chat()

    def query(self, question: str, session_id: Optional[str] = None, *, rag_mode: bool = True) -> RAGResponse:
        """Non-streaming full-graph run returning a structured response."""
        session_id = self._ensure_session(session_id)
        start = time.perf_counter()
        state = self._get_graph().invoke(
            {"question": question, "session_id": session_id, "rag_mode": rag_mode}
        )
        return self._build_response(state, session_id, time.perf_counter() - start)

    def stream_query(
        self,
        question: str,
        session_id: Optional[str] = None,
        *,
        rag_mode: bool = True,
    ) -> tuple[Iterator[str], RAGResponse]:
        """Stream answer tokens; the returned RAGResponse is filled as the run completes."""
        session_id = self._ensure_session(session_id)
        graph = self._get_graph()
        response = RAGResponse(
            session_id=session_id,
            provider=self.llm_settings.provider,
            model=self.llm_settings.model,
        )
        inputs = {"question": question, "session_id": session_id, "rag_mode": rag_mode}

        def generator() -> Iterator[str]:
            start = time.perf_counter()
            token_parts: list[str] = []
            final_state: dict = {}
            for mode, chunk in graph.stream(inputs, stream_mode=["values", "messages"]):
                if mode == "messages":
                    message, meta = chunk
                    if meta.get("langgraph_node") == "generate":
                        text = _text(message.content)
                        if text:
                            token_parts.append(text)
                            yield text
                elif mode == "values":
                    final_state = chunk
                    if chunk.get("sources"):
                        response.source_documents = [
                            SourceDocument(**s) for s in chunk["sources"] if isinstance(s, dict)
                        ]
            self._fill_response(response, final_state, "".join(token_parts), time.perf_counter() - start)

        return generator(), response

    # -- Response assembly ---------------------------------------------
    def _build_response(self, state: dict, session_id: str, latency: float) -> RAGResponse:
        response = RAGResponse(
            session_id=session_id,
            provider=self.llm_settings.provider,
            model=self.llm_settings.model,
        )
        self._fill_response(response, state, state.get("answer", ""), latency)
        return response

    def _fill_response(self, response: RAGResponse, state: dict, answer: str, latency: float) -> None:
        response.answer = state.get("answer") or answer
        response.source_documents = [
            SourceDocument(**s) for s in state.get("sources", []) if isinstance(s, dict)
        ]
        response.retrieved_chunks = len(state.get("documents", []))
        response.latency_seconds = round(latency, 3)
        response.metadata = state.get("debug", {})
        usage = (state.get("debug", {}) or {}).get("token_usage") or {}
        if usage:
            response.token_usage = TokenUsage(
                input_tokens=usage.get("input_tokens", 0),
                output_tokens=usage.get("output_tokens", 0),
                total_tokens=usage.get("total_tokens", 0),
            )


def _text(content) -> str:
    if isinstance(content, list):
        return "".join(
            part.get("text", "") if isinstance(part, dict) else str(part) for part in content
        )
    return str(content or "")
