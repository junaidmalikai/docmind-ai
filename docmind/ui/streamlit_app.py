"""DocMind AI — Streamlit entry point (application layer)."""

from __future__ import annotations

import streamlit as st

from docmind.ui.chat import chat_panel_html, handle_user_query
from docmind.ui.session import get_rag_service, init_session_state, is_api_ready
from docmind.ui.sidebar import render_sidebar
from docmind.ui.theme import inject_theme, render_footer


def main() -> None:
    st.set_page_config(
        page_title="DocMind AI",
        page_icon="📄",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_theme()

    rag = get_rag_service()
    rag.warm_up_async()  # UI renders immediately; heavy models load in the background.
    init_session_state(rag)
    render_sidebar(rag)

    status = "API ready" if is_api_ready() else "API key required"
    st.markdown(
        f"""
        <div class="dm-header">
          <h1>DocMind AI</h1>
          <p>{st.session_state.provider} · {'RAG on' if st.session_state.rag_mode else 'General chat'} · {status}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not is_api_ready():
        st.info(
            "Enter your API key in the sidebar — it is checked **automatically**. "
            "Upload unlocks only after the key is valid."
        )
    elif not rag.list_documents():
        st.info(
            "Upload and **index** at least one document in the sidebar before you can chat."
        )

    chat_slot = st.empty()
    chat_slot.markdown(chat_panel_html(st.session_state.messages), unsafe_allow_html=True)

    api_ok = is_api_ready()
    has_docs = bool(rag.list_documents())

    if api_ok and has_docs:
        prompt = st.chat_input("Ask something about your documents…")
        if prompt:
            handle_user_query(rag, prompt, chat_slot)
    elif not api_ok:
        st.chat_input("Validate API key to start chatting…", disabled=True)
    else:
        st.chat_input("Index a document before chatting…", disabled=True)

    render_footer()


if __name__ == "__main__":
    main()
