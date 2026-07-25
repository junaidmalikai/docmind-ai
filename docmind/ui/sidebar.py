"""Sidebar: LLM settings, chats, uploads, and knowledge base."""

from __future__ import annotations

import streamlit as st

from docmind.config.providers import PROVIDER_DEFAULTS, provider_settings
from docmind.services.loaders import SUPPORTED_EXTENSIONS
from docmind.services.rag_service import RAGService
from docmind.ui.session import (
    apply_rag_params,
    auto_check_api_key,
    is_api_ready,
    load_chat,
    on_credentials_widget_change,
    run_api_key_check,
    start_new_chat,
)
from docmind.utils.api_validation import validate_upload_file

SUPPORTED_UPLOAD_TYPES = sorted(ext.lstrip(".") for ext in SUPPORTED_EXTENSIONS)
ALLOWED_EXT_SET = set(SUPPORTED_EXTENSIONS)


def _render_chats_section(rag: RAGService) -> None:
    st.subheader("Chats")
    if st.button("New chat", use_container_width=True, key="new_chat_btn"):
        start_new_chat(rag)
        st.rerun()

    st.text_input(
        "Search",
        key="chat_search",
        placeholder="Search chats…",
        help="Filter chat history by title.",
    )
    search = (st.session_state.get("chat_search") or "").strip()
    sessions = rag.list_chat_sessions(search=search)

    if not sessions:
        st.caption("No chats found." if search else "No chats yet.")
        return

    for session in sessions[:30]:
        label = session.title or "Untitled"
        is_active = session.session_id == st.session_state.session_id
        left, right = st.columns([0.84, 0.16], gap="small")
        with left:
            if st.button(
                f"{'● ' if is_active else ''}{label}",
                key=f"chat_{session.session_id}",
                use_container_width=True,
            ):
                load_chat(rag, session.session_id)
                st.rerun()
        with right:
            if st.button("×", key=f"rm_{session.session_id}", help="Delete chat", use_container_width=True):
                rag.delete_chat(session.session_id)
                if st.session_state.session_id == session.session_id:
                    st.session_state.session_id = None
                    st.session_state.messages = []
                st.rerun()


def _render_upload_section(rag: RAGService) -> None:
    st.subheader("Upload documents")

    if not is_api_ready():
        st.caption("Locked until API key is valid")
        st.file_uploader(
            "Upload disabled",
            type=SUPPORTED_UPLOAD_TYPES,
            accept_multiple_files=True,
            disabled=True,
            key="uploader_locked",
            label_visibility="collapsed",
        )
        return

    uploads = st.file_uploader(
        "Supported: PDF, DOCX, TXT, MD, CSV, XLSX, XLS (max 25 MB each)",
        type=SUPPORTED_UPLOAD_TYPES,
        accept_multiple_files=True,
        key="uploader_ready",
    )
    if uploads and st.button("Index files", type="primary", use_container_width=True):
        if not is_api_ready():
            st.error("API key is no longer valid.")
            return
        apply_rag_params(rag)
        progress = st.progress(0.0)
        any_ok = False
        for i, uploaded in enumerate(uploads):
            data = uploaded.getvalue()
            ok, msg = validate_upload_file(uploaded.name, data, ALLOWED_EXT_SET)
            if not ok:
                st.error(msg)
            else:
                try:
                    document, n_chunks = rag.ingest_bytes(uploaded.name, data)
                    if n_chunks <= 0:
                        st.warning(f"{document.filename}: 0 chunks extracted.")
                    else:
                        st.success(f"{document.filename} → {n_chunks} chunks")
                        any_ok = True
                except Exception as exc:
                    st.error(f"{uploaded.name}: {exc}")
            progress.progress((i + 1) / len(uploads))
        if any_ok:
            st.rerun()


def _render_knowledge_base(rag: RAGService) -> None:
    st.subheader("Knowledge base")
    stats = rag.get_document_analytics()
    c1, c2 = st.columns(2)
    c1.metric("Documents", stats["total_documents"])
    c2.metric("Chunks", stats["total_chunks"])

    docs = rag.list_documents()
    if not docs:
        st.caption("No documents indexed yet.")
        return

    for doc in docs:
        left, right = st.columns([0.84, 0.16], gap="small")
        short = doc.filename if len(doc.filename) <= 18 else doc.filename[:15] + "…"
        with left:
            st.markdown(f"**{doc.filename}**")
            st.caption(f"{doc.file_type} · {short}")
        with right:
            if st.button("×", key=f"del_{doc.id}", help="Delete document", use_container_width=True):
                rag.delete_document(doc.id)
                st.rerun()


def render_sidebar(rag: RAGService) -> None:
    with st.sidebar:
        st.title("DocMind AI")
        st.caption("Chat with your documents")

        st.subheader("LLM settings")
        providers = list(PROVIDER_DEFAULTS.keys())
        provider = st.selectbox(
            "Provider",
            providers,
            index=providers.index(st.session_state.provider)
            if st.session_state.provider in providers
            else 0,
            key="provider_select",
            on_change=on_credentials_widget_change,
        )
        if provider != st.session_state.provider:
            defaults = provider_settings(provider)
            st.session_state.provider = provider
            st.session_state.model = defaults["model"]
            st.session_state.api_key = defaults["api_key"]
            st.session_state.base_url = defaults["base_url"]
            st.session_state.checked_fingerprint = ""
            st.session_state.api_key_valid = False
            st.session_state.validated_fingerprint = ""
            st.session_state.api_key_message = "Checking API key…"
            st.rerun()

        st.text_input(
            "API key",
            type="password",
            key="api_key",
            help="Checked automatically when you paste or change it.",
            on_change=on_credentials_widget_change,
        )
        if st.session_state.provider == "Custom OpenAI Compatible Endpoint":
            st.text_input("Base URL", key="base_url", on_change=on_credentials_widget_change)

        st.slider("Temperature", 0.0, 1.0, key="temperature", step=0.05)
        st.toggle("RAG mode (use documents)", key="rag_mode")

        st.subheader("RAG settings")
        st.number_input(
            "Chunk size",
            min_value=100,
            max_value=4000,
            step=50,
            key="chunk_size",
            help="Characters per chunk when indexing new files. Existing files keep their old chunks.",
        )
        st.number_input(
            "Chunk overlap",
            min_value=0,
            max_value=2000,
            step=25,
            key="chunk_overlap",
            help="Overlap between consecutive chunks. Must be smaller than chunk size.",
        )
        st.number_input(
            "Retrieve chunks (K)",
            min_value=1,
            max_value=20,
            step=1,
            key="max_results",
            help="How many document chunks to retrieve per question.",
        )
        apply_rag_params(rag)

        auto_check_api_key(rag)

        if st.button("Re-check API key", use_container_width=True):
            with st.spinner("Re-checking API key…"):
                ok, msg = run_api_key_check(rag, force=True)
            st.success(msg) if ok else st.error(msg)
            st.rerun()

        if is_api_ready():
            st.success("API key valid — upload & chat unlocked")
        elif (st.session_state.api_key or "").strip():
            st.error(st.session_state.api_key_message)
        else:
            st.info("Waiting for API key…")

        st.divider()
        _render_chats_section(rag)
        st.divider()
        _render_upload_section(rag)
        st.divider()
        _render_knowledge_base(rag)
