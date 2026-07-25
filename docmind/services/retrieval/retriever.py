"""Retrieval orchestration built on the official ``VectorStoreRetriever``.

The base retriever is Chroma's ``as_retriever``. Optional query expansion and
reranking are layered on top using official APIs (multi-query union + Chroma
similarity scores) rather than hand-rolled search.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from docmind.config.logging import get_logger

if TYPE_CHECKING:
    from langchain_core.documents import Document

    from docmind.services.vectorstore import VectorStoreService

logger = get_logger(__name__)


class RetrievalService:
    """Wraps a vector store with an official retriever and helper compositions."""

    def __init__(self, vector_store: "VectorStoreService", k: int):
        self.vector_store = vector_store
        self.k = k

    def retrieve(self, queries: list[str]) -> list["Document"]:
        """Retrieve and de-duplicate documents for one or more queries."""
        retriever = self.vector_store.as_retriever(self.k)
        seen: set[str] = set()
        merged: list["Document"] = []
        for query in queries:
            query = (query or "").strip()
            if not query:
                continue
            for doc in retriever.invoke(query):
                key = doc.metadata.get("chunk_id") or doc.page_content[:64]
                if key in seen:
                    continue
                seen.add(key)
                merged.append(doc)
        return merged

    def rerank_by_score(self, question: str, documents: list["Document"]) -> list["Document"]:
        """Reorder documents by Chroma similarity score and keep the top-k."""
        if not documents:
            return documents
        scored = self.vector_store.similarity_search_with_score(question, k=max(self.k, len(documents)))
        order = {
            doc.metadata.get("chunk_id"): score
            for doc, score in scored
            if doc.metadata.get("chunk_id")
        }
        ranked = sorted(
            documents,
            key=lambda d: order.get(d.metadata.get("chunk_id"), 1e9),
        )
        return ranked[: self.k]
