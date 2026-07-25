"""Chat rendering — scrollable panel; header/input stay fixed."""

from __future__ import annotations

import html

import streamlit as st

from docmind.services.llm import LLMConfigurationError
from docmind.services.rag_service import RAGService
from docmind.ui.session import apply_llm_settings, apply_rag_params, is_api_ready

USER_AVATAR = "👤"
AI_AVATAR = "🤖"


def bubble_html(role: str, content: str, sources: list[str] | None = None) -> str:
    safe = html.escape(content)
    if role == "user":
        return (
            '<div class="dm-row dm-row-user">'
            f'<div class="dm-bubble dm-bubble-user">{safe}</div>'
            f'<div class="dm-avatar dm-avatar-user">{USER_AVATAR}</div>'
            "</div>"
        )
    sources_html = ""
    if sources:
        joined = html.escape(", ".join(sorted(set(sources))))
        sources_html = f'<div class="dm-sources">Sources: {joined}</div>'
    return (
        '<div class="dm-row dm-row-ai">'
        f'<div class="dm-avatar dm-avatar-ai">{AI_AVATAR}</div>'
        f'<div class="dm-bubble dm-bubble-ai">{safe}{sources_html}</div>'
        "</div>"
    )


def chat_panel_html(messages: list[dict], live_html: str = "") -> str:
    """One scrollable panel; newest messages stay in view via column-reverse."""
    parts = ['<div class="dm-chat" id="dm-chat-scroll">']
    if live_html:
        parts.append(live_html)
    for message in reversed(messages):
        parts.append(bubble_html(message["role"], message["content"], message.get("sources")))
    if not messages and not live_html:
        parts.append(
            '<div class="dm-empty">Ask a question about your indexed documents.</div>'
        )
    parts.append("</div>")
    return "".join(parts)


def handle_user_query(rag: RAGService, prompt: str, chat_slot) -> None:
    prompt = (prompt or "").strip()
    if not prompt:
        st.warning("Please enter a non-empty question.")
        return
    if len(prompt) > 8000:
        st.warning("Question is too long (max 8000 characters).")
        return
    if not is_api_ready():
        st.error("API key is not valid yet. Paste a valid key in the sidebar.")
        return

    apply_llm_settings(rag)
    apply_rag_params(rag)

    if not rag.list_documents():
        st.warning("Upload and index at least one document before chatting.")
        return

    st.session_state.messages.append({"role": "user", "content": prompt})
    history = st.session_state.messages
    chat_slot.markdown(chat_panel_html(history), unsafe_allow_html=True)

    answer = ""
    source_names: list[str] = []
    try:
        token_stream, response = rag.stream_query(
            prompt,
            session_id=st.session_state.session_id,
            rag_mode=st.session_state.rag_mode,
        )
        st.session_state.session_id = response.session_id

        collected: list[str] = []
        for token in token_stream:
            if not token:
                continue
            collected.append(token)
            source_names = sorted({s.name for s in response.source_documents})
            chat_slot.markdown(
                chat_panel_html(
                    history,
                    bubble_html("assistant", "".join(collected), source_names),
                ),
                unsafe_allow_html=True,
            )
        answer = "".join(collected) or response.answer
        source_names = sorted({s.name for s in response.source_documents})
        if not answer.strip():
            answer = "Sorry, I could not generate a response. Please try again."
            chat_slot.markdown(
                chat_panel_html(history, bubble_html("assistant", answer, source_names)),
                unsafe_allow_html=True,
            )
    except LLMConfigurationError as exc:
        st.error(str(exc))
        st.session_state.messages.pop()
        return
    except Exception as exc:
        st.error(f"Chat failed: {exc}")
        st.session_state.messages.pop()
        return

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": source_names}
    )
    st.rerun()
