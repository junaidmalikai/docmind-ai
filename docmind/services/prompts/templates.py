"""Prompt templates as official ``ChatPromptTemplate`` objects.

Prompts are never concatenated by hand: conversation memory is injected through a
``MessagesPlaceholder`` and inputs are passed as template variables.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # langchain_core transitively loads torch; keep it out of startup
    from langchain_core.prompts import ChatPromptTemplate

RAG_SYSTEM_PROMPT = """You are DocMind AI, an enterprise document-grounded RAG assistant.

You answer strictly from the retrieved document context.

Rules:
- Answer only from the provided document context.
- If the answer is not present in the context, reply politely:
  "Sorry, I couldn't find this information in your uploaded knowledge base. I only answer using your uploaded PDFs, documents, spreadsheets, images, and other indexed files."
- Do not hallucinate or invent information.
- Cite sources inline using the labels provided in context, for example: (Source 1).
- Keep the response clear, useful, and concise.
- Do not invent page numbers, filenames, or facts."""

GENERAL_SYSTEM_PROMPT = """You are a helpful AI assistant.

Rules:
- Be clear, accurate, and concise.
- Use markdown formatting when helpful.
- For code, use fenced code blocks with language tags."""

TITLE_SYSTEM_PROMPT = """Generate a short conversation title from the user's first message.
Return ONLY the title, maximum 5 words, no quotes, no trailing punctuation.
Examples: Company Financial Report, Invoice OCR, Resume Review, AI Research Paper"""

QUERY_EXPANSION_SYSTEM_PROMPT = """You rewrite a user question into diverse search queries for document retrieval.
Return 3 alternative queries, one per line, no numbering, no extra text."""


def build_rag_prompt() -> "ChatPromptTemplate":
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

    return ChatPromptTemplate.from_messages(
        [
            ("system", RAG_SYSTEM_PROMPT),
            MessagesPlaceholder("chat_history", optional=True),
            ("human", "Document context:\n{context}\n\nQuestion: {question}"),
        ]
    )


def build_general_prompt() -> "ChatPromptTemplate":
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

    return ChatPromptTemplate.from_messages(
        [
            ("system", GENERAL_SYSTEM_PROMPT),
            MessagesPlaceholder("chat_history", optional=True),
            ("human", "{question}"),
        ]
    )


def build_title_prompt() -> "ChatPromptTemplate":
    from langchain_core.prompts import ChatPromptTemplate

    return ChatPromptTemplate.from_messages(
        [
            ("system", TITLE_SYSTEM_PROMPT),
            ("human", "{question}"),
        ]
    )


def build_query_expansion_prompt() -> "ChatPromptTemplate":
    from langchain_core.prompts import ChatPromptTemplate

    return ChatPromptTemplate.from_messages(
        [
            ("system", QUERY_EXPANSION_SYSTEM_PROMPT),
            ("human", "{question}"),
        ]
    )
