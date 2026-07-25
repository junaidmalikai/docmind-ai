"""LLM provider defaults shared across the UI and services."""

from __future__ import annotations

from docmind.config.settings import settings

PROVIDER_DEFAULTS: dict[str, dict[str, str]] = {
    "Groq": {
        "model": settings.GROQ_MODEL,
        "api_key": settings.GROQ_API_KEY,
        "base_url": "https://api.groq.com/openai/v1",
    },
    "OpenAI": {
        "model": settings.OPENAI_MODEL,
        "api_key": settings.OPENAI_API_KEY,
        "base_url": "https://api.openai.com/v1",
    },
    "Claude": {
        "model": settings.CLAUDE_MODEL,
        "api_key": settings.ANTHROPIC_API_KEY,
        "base_url": "",
    },
    "Gemini": {
        "model": settings.GEMINI_MODEL,
        "api_key": settings.GEMINI_API_KEY,
        "base_url": "",
    },
    "Custom OpenAI Compatible Endpoint": {
        "model": settings.CUSTOM_MODEL,
        "api_key": settings.CUSTOM_API_KEY,
        "base_url": settings.CUSTOM_BASE_URL,
    },
}

DEFAULT_PROVIDER = "Groq"


def resolve_provider(provider: str | None) -> str:
    if provider in PROVIDER_DEFAULTS:
        return provider
    return DEFAULT_PROVIDER


def provider_settings(provider: str) -> dict[str, str]:
    return PROVIDER_DEFAULTS.get(provider, PROVIDER_DEFAULTS[DEFAULT_PROVIDER])


def resolve_model(provider: str, custom_model: str | None = None) -> str:
    """Use a client override when provided, otherwise the built-in default."""
    custom = (custom_model or "").strip()
    if custom:
        return custom
    return provider_settings(provider)["model"]
