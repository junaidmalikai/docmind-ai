"""Pydantic models describing the public data contract of the app."""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class LLMSettings(BaseModel):
    """Runtime configuration for the active chat model."""

    provider: str
    model: str
    api_key: str = ""
    base_url: str = ""
    temperature: float = 0.7
    top_p: float = 0.9
    max_tokens: int = 2048
    streaming: bool = True


class SourceDocument(BaseModel):
    """A citation surfaced alongside an answer."""

    document_id: str = ""
    name: str
    link: str = ""
    citation: str = ""


class DocumentInfo(BaseModel):
    """Metadata for an indexed source document."""

    id: str
    filename: str
    file_type: str
    source_path: str = ""
    chunk_count: int = 0
    created_at: Optional[datetime] = None


class ChatMessage(BaseModel):
    """A single stored chat message."""

    role: str
    content: str
    created_at: Optional[datetime] = None
    sources: list[SourceDocument] = Field(default_factory=list)


class ChatSessionSummary(BaseModel):
    """Sidebar summary for a saved conversation."""

    session_id: str
    title: str
    updated_at: Optional[datetime] = None
    message_count: int = 0
    provider: str = "Groq"
    model: str = ""
    created_at: Optional[datetime] = None


class TokenUsage(BaseModel):
    """Token accounting when the provider reports it."""

    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0


class RAGResponse(BaseModel):
    """Structured result returned by the RAG pipeline."""

    answer: str = ""
    source_documents: list[SourceDocument] = Field(default_factory=list)
    session_id: str = ""
    provider: str = ""
    model: str = ""
    retrieved_chunks: int = 0
    latency_seconds: float = 0.0
    token_usage: TokenUsage = Field(default_factory=TokenUsage)
    metadata: dict = Field(default_factory=dict)
