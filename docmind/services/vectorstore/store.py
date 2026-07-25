"""Thin accessor around the official ``langchain_chroma.Chroma`` store.

We do not wrap or reimplement Chroma's behavior; this class only owns lazy
construction (so app startup stays fast), caches the single instance, and exposes
``add_documents`` / ``as_retriever`` / ``delete`` through the official API.
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING, Optional

from docmind.config.logging import get_logger
from docmind.config.settings import settings
from docmind.services.embeddings import create_embeddings
from docmind.utils.helpers import ensure_dir

if TYPE_CHECKING:
    from langchain_chroma import Chroma
    from langchain_core.documents import Document
    from langchain_core.vectorstores import VectorStoreRetriever

logger = get_logger(__name__)

COLLECTION_NAME = "docmind_chunks"


class VectorStoreService:
    """Owns a single, lazily-created Chroma vector store."""

    def __init__(
        self,
        persist_directory: Optional[str] = None,
        embedding_provider: Optional[str] = None,
        embedding_model: Optional[str] = None,
    ):
        self.persist_directory = persist_directory or settings.CHROMA_PATH
        self.embedding_provider = embedding_provider or settings.EMBEDDING_PROVIDER
        self.embedding_model = embedding_model or settings.EMBEDDING_MODEL
        self._store: Optional["Chroma"] = None
        self._lock = threading.Lock()

    @property
    def store(self) -> "Chroma":
        if self._store is None:
            with self._lock:
                if self._store is None:
                    self._store = self._build_store()
        return self._store

    @property
    def is_loaded(self) -> bool:
        return self._store is not None

    def _build_store(self) -> "Chroma":
        from langchain_chroma import Chroma

        ensure_dir(self.persist_directory)
        embeddings = create_embeddings(self.embedding_provider, self.embedding_model)
        logger.info("Opening Chroma at %s", self.persist_directory)
        return Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=embeddings,
            persist_directory=self.persist_directory,
        )

    def warm_up(self) -> None:
        """Force the embedding model + Chroma client to load (background friendly)."""
        _ = self.store
        create_embeddings(self.embedding_provider, self.embedding_model).embed_query("warm up")

    def add_documents(self, documents: list["Document"]) -> int:
        if not documents:
            return 0
        ids = [doc.metadata["chunk_id"] for doc in documents]
        self.store.add_documents(documents=documents, ids=ids)
        return len(documents)

    def delete_document(self, document_id: str) -> None:
        """Delete every chunk belonging to a document via the official API."""
        existing = self.store.get(where={"document_id": document_id})
        ids = existing.get("ids") or []
        if ids:
            self.store.delete(ids=ids)

    def as_retriever(self, k: int) -> "VectorStoreRetriever":
        return self.store.as_retriever(search_kwargs={"k": k})

    def similarity_search_with_score(self, query: str, k: int):
        return self.store.similarity_search_with_score(query, k=k)
