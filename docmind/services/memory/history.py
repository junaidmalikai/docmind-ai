"""Chat memory using the official ``SQLChatMessageHistory``.

Messages (and their source citations, stored in ``additional_kwargs``) are
persisted by LangChain's SQL-backed history. No custom memory abstraction is
introduced; this service only adapts it to the app's typed models.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from docmind.models import ChatMessage, SourceDocument

if TYPE_CHECKING:
    from langchain_community.chat_message_histories import SQLChatMessageHistory
    from langchain_core.messages import BaseMessage

TABLE_NAME = "chat_messages"
_SOURCES_KEY = "docmind_sources"


def _text(content) -> str:
    if isinstance(content, list):
        return "".join(
            part.get("text", "") if isinstance(part, dict) else str(part) for part in content
        )
    return str(content or "")


class ChatMemoryService:
    """Thin adapter over ``SQLChatMessageHistory``."""

    def __init__(self, connection: str, max_history: int):
        self.connection = connection
        self.max_history = max_history

    def history(self, session_id: str) -> "SQLChatMessageHistory":
        from langchain_community.chat_message_histories import SQLChatMessageHistory

        return SQLChatMessageHistory(
            session_id=session_id,
            connection=self.connection,
            table_name=TABLE_NAME,
        )

    def recent_messages(self, session_id: str) -> list["BaseMessage"]:
        """Last ``max_history`` turns as LangChain messages for the prompt placeholder."""
        messages = self.history(session_id).messages
        if self.max_history and len(messages) > self.max_history * 2:
            return messages[-self.max_history * 2 :]
        return messages

    def add_exchange(
        self,
        session_id: str,
        question: str,
        answer: str,
        sources: list[dict],
    ) -> None:
        from langchain_core.messages import AIMessage, HumanMessage

        history = self.history(session_id)
        history.add_messages(
            [
                HumanMessage(content=question),
                AIMessage(content=answer, additional_kwargs={_SOURCES_KEY: sources or []}),
            ]
        )

    def load_chat_messages(self, session_id: str) -> list[ChatMessage]:
        out: list[ChatMessage] = []
        for message in self.history(session_id).messages:
            role = "user" if message.type == "human" else "assistant"
            raw_sources = (message.additional_kwargs or {}).get(_SOURCES_KEY, []) or []
            sources = [SourceDocument(**s) for s in raw_sources if isinstance(s, dict)]
            out.append(ChatMessage(role=role, content=_text(message.content), sources=sources))
        return out

    def message_count(self, session_id: str) -> int:
        return len(self.history(session_id).messages)

    def clear(self, session_id: str) -> None:
        self.history(session_id).clear()
