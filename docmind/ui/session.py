"""Streamlit session-state, API-key validation and RAG service wiring."""

from __future__ import annotations

import streamlit as st

from docmind.config.providers import provider_settings, resolve_provider
from docmind.config.settings import settings
from docmind.models import LLMSettings
from docmind.services.rag_service import RAGService
from docmind.utils.api_validation import (
    credentials_fingerprint,
    validate_provider_api_key,
    validate_settings_form,
)


@st.cache_resource(show_spinner="Starting DocMind AI…")
def get_rag_service() -> RAGService:
    provider = resolve_provider(settings.DEFAULT_PROVIDER)
    defaults = provider_settings(provider)
    llm_settings = LLMSettings(
        provider=provider,
        model=defaults["model"],
        api_key=defaults["api_key"],
        base_url=defaults["base_url"],
        temperature=settings.TEMPERATURE,
        top_p=settings.TOP_P,
        max_tokens=settings.MAX_TOKENS,
        streaming=True,
    )
    return RAGService(settings, llm_settings=llm_settings)


def init_session_state(rag: RAGService) -> None:
    defaults = {
        "session_id": None,
        "messages": [],
        "provider": rag.llm_settings.provider,
        "model": rag.llm_settings.model,
        "api_key": rag.llm_settings.api_key or "",
        "base_url": rag.llm_settings.base_url or "",
        "temperature": min(1.0, float(settings.TEMPERATURE)),
        "rag_mode": True,
        "chunk_size": int(settings.CHUNK_SIZE),
        "chunk_overlap": int(settings.CHUNK_OVERLAP),
        "max_results": int(settings.MAX_RESULTS),
        "api_key_valid": False,
        "api_key_message": "Paste an API key — it will be checked automatically.",
        "validated_fingerprint": "",
        "checked_fingerprint": "",
        "chat_search": "",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def apply_rag_params(rag: RAGService) -> None:
    """Push sidebar RAG knobs into the live service."""
    chunk_size = int(st.session_state.get("chunk_size", settings.CHUNK_SIZE))
    chunk_overlap = int(st.session_state.get("chunk_overlap", settings.CHUNK_OVERLAP))
    max_results = int(st.session_state.get("max_results", settings.MAX_RESULTS))
    # Keep overlap valid if the user dragged chunk size down.
    if chunk_overlap >= chunk_size:
        chunk_overlap = max(0, chunk_size // 5)
        st.session_state.chunk_overlap = chunk_overlap
    rag.update_rag_params(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        max_results=max_results,
    )


def current_fingerprint() -> str:
    return credentials_fingerprint(
        st.session_state.provider,
        st.session_state.api_key,
        st.session_state.model,
        st.session_state.get("base_url", ""),
    )


def is_api_ready() -> bool:
    return bool(st.session_state.api_key_valid) and (
        st.session_state.validated_fingerprint == current_fingerprint()
    )


def apply_llm_settings(rag: RAGService) -> None:
    model = (st.session_state.model or "").strip() or provider_settings(
        st.session_state.provider
    )["model"]
    st.session_state.model = model
    rag.update_llm(
        LLMSettings(
            provider=st.session_state.provider,
            model=model,
            api_key=st.session_state.api_key.strip(),
            base_url=st.session_state.base_url.strip(),
            temperature=float(st.session_state.temperature),
            top_p=settings.TOP_P,
            max_tokens=settings.MAX_TOKENS,
            streaming=True,
        )
    )


@st.cache_data(ttl=300, show_spinner=False)
def _cached_key_check(
    fingerprint: str,
    provider: str,
    api_key: str,
    model: str,
    base_url: str,
) -> tuple[bool, str]:
    _ = fingerprint
    return validate_provider_api_key(provider, api_key, base_url=base_url, model=model)


def run_api_key_check(rag: RAGService, *, force: bool = False) -> tuple[bool, str]:
    provider = st.session_state.provider
    api_key = (st.session_state.api_key or "").strip()
    model = (st.session_state.model or "").strip() or provider_settings(provider)["model"]
    st.session_state.model = model
    base_url = (st.session_state.get("base_url") or "").strip()
    fp = current_fingerprint()

    if not force and fp == st.session_state.get("checked_fingerprint"):
        return st.session_state.api_key_valid, st.session_state.api_key_message

    if not api_key:
        st.session_state.api_key_valid = False
        st.session_state.api_key_message = "Paste an API key — it will be checked automatically."
        st.session_state.validated_fingerprint = ""
        st.session_state.checked_fingerprint = fp
        return False, st.session_state.api_key_message

    ok, msg = validate_settings_form(provider, api_key, model, base_url=base_url)
    if not ok:
        st.session_state.api_key_valid = False
        st.session_state.api_key_message = msg
        st.session_state.validated_fingerprint = ""
        st.session_state.checked_fingerprint = fp
        return False, msg

    if force:
        _cached_key_check.clear()

    ok, msg = _cached_key_check(fp, provider, api_key, model, base_url)
    st.session_state.checked_fingerprint = fp
    st.session_state.api_key_message = msg

    if ok:
        apply_llm_settings(rag)
        st.session_state.api_key_valid = True
        st.session_state.validated_fingerprint = fp
    else:
        st.session_state.api_key_valid = False
        st.session_state.validated_fingerprint = ""

    return ok, msg


def auto_check_api_key(rag: RAGService) -> None:
    if current_fingerprint() == st.session_state.get("checked_fingerprint"):
        return
    with st.spinner("Checking API key…"):
        run_api_key_check(rag, force=False)


def on_credentials_widget_change() -> None:
    st.session_state.checked_fingerprint = ""
    st.session_state.api_key_valid = False
    st.session_state.validated_fingerprint = ""
    st.session_state.api_key_message = "Checking API key…"


def start_new_chat(rag: RAGService) -> None:
    st.session_state.session_id = rag.create_chat()
    st.session_state.messages = []


def load_chat(rag: RAGService, session_id: str) -> None:
    st.session_state.session_id = session_id
    loaded = []
    for msg in rag.load_chat_messages(session_id):
        entry = {"role": msg.role, "content": msg.content}
        if msg.sources:
            entry["sources"] = [s.name for s in msg.sources]
        loaded.append(entry)
    st.session_state.messages = loaded
