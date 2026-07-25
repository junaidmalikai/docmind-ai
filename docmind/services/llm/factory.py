"""Build official LangChain chat models per provider.

Returns bare ``BaseChatModel`` instances — no provider-specific wrapper classes.
Each provider uses its dedicated official integration package.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from docmind.models import LLMSettings

if TYPE_CHECKING:
    from langchain_core.language_models.chat_models import BaseChatModel


class LLMConfigurationError(ValueError):
    """Raised when provider credentials or settings are missing/invalid."""


def _require(value: str, message: str) -> str:
    cleaned = (value or "").strip()
    if not cleaned:
        raise LLMConfigurationError(message)
    return cleaned


def create_chat_model(settings: LLMSettings) -> "BaseChatModel":
    """Construct the official chat model for the selected provider."""
    provider = settings.provider

    if provider == "Groq":
        from langchain_groq import ChatGroq

        return ChatGroq(
            model=settings.model,
            api_key=_require(settings.api_key, "Missing API key for Groq."),
            temperature=settings.temperature,
            max_tokens=settings.max_tokens,
            streaming=settings.streaming,
        )

    if provider in {"OpenAI", "Custom OpenAI Compatible Endpoint"}:
        from langchain_openai import ChatOpenAI

        if provider == "Custom OpenAI Compatible Endpoint":
            base_url = _require(
                settings.base_url, "Missing base URL for the custom endpoint."
            ).rstrip("/")
        else:
            base_url = (settings.base_url or "https://api.openai.com/v1").rstrip("/")

        return ChatOpenAI(
            model=settings.model,
            api_key=_require(settings.api_key, f"Missing API key for {provider}."),
            base_url=base_url,
            temperature=settings.temperature,
            top_p=settings.top_p,
            max_tokens=settings.max_tokens,
            streaming=settings.streaming,
        )

    if provider == "Claude":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(
            model=settings.model,
            api_key=_require(settings.api_key, "Missing API key for Claude."),
            temperature=settings.temperature,
            top_p=settings.top_p,
            max_tokens=settings.max_tokens,
            streaming=settings.streaming,
        )

    if provider == "Gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=settings.model,
            google_api_key=_require(settings.api_key, "Missing API key for Gemini."),
            temperature=settings.temperature,
            top_p=settings.top_p,
            max_output_tokens=settings.max_tokens,
        )

    raise LLMConfigurationError(f"Unsupported provider: {provider}")
