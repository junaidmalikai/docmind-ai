"""Validate LLM provider API keys and uploads before use.

This stays custom on purpose — API-key validation and the file-upload interface
are explicitly allowed to remain application code.
"""

from __future__ import annotations

import re
from pathlib import Path

import httpx

# Soft format hints — not a substitute for a live provider check.
_KEY_PREFIX_HINTS: dict[str, tuple[str, ...]] = {
    "OpenAI": ("sk-", "sk-proj-"),
    "Groq": ("gsk_",),
    "Claude": ("sk-ant-",),
    "Gemini": ("AIza",),
}

MAX_UPLOAD_BYTES = 25 * 1024 * 1024  # 25 MB per file
MIN_KEY_LENGTH = 20


def validate_settings_form(
    provider: str,
    api_key: str,
    model: str,
    *,
    base_url: str = "",
) -> tuple[bool, str]:
    """Local checks before hitting the network."""
    key = (api_key or "").strip()
    model_name = (model or "").strip()
    url = (base_url or "").strip()

    if not provider:
        return False, "Select an LLM provider."
    if not model_name:
        return False, "Model name is required."
    if len(model_name) > 120:
        return False, "Model name is too long."
    if not key:
        return False, "API key is required. Paste a valid API key."
    if len(key) < MIN_KEY_LENGTH:
        return False, f"API key looks too short (min {MIN_KEY_LENGTH} characters)."
    if any(ch.isspace() for ch in key):
        return False, "API key must not contain spaces. Remove spaces and try again."

    hints = _KEY_PREFIX_HINTS.get(provider)
    if hints and not any(key.startswith(p) for p in hints):
        expected = " or ".join(f"`{p}`" for p in hints)
        return False, f"API key format looks wrong for {provider}. Expected prefix {expected}."

    if provider == "Custom OpenAI Compatible Endpoint":
        if not url:
            return False, "Base URL is required for a custom endpoint."
        if not (url.startswith("http://") or url.startswith("https://")):
            return False, "Base URL must start with http:// or https://"

    return True, "Settings look OK."


def validate_provider_api_key(
    provider: str,
    api_key: str,
    *,
    base_url: str = "",
    model: str = "",
) -> tuple[bool, str]:
    """Local format checks + a live provider auth ping. Returns (ok, message)."""
    ok, msg = validate_settings_form(provider, api_key, model or "model", base_url=base_url)
    if not ok:
        return False, msg

    key = (api_key or "").strip()

    try:
        with httpx.Client(timeout=20.0) as client:
            if provider in {"Groq", "OpenAI", "Custom OpenAI Compatible Endpoint"}:
                url = (base_url or "").rstrip("/") or (
                    "https://api.groq.com/openai/v1"
                    if provider == "Groq"
                    else "https://api.openai.com/v1"
                )
                response = client.get(
                    f"{url}/models",
                    headers={"Authorization": f"Bearer {key}"},
                )
            elif provider == "Claude":
                response = client.get(
                    "https://api.anthropic.com/v1/models",
                    headers={"x-api-key": key, "anthropic-version": "2023-06-01"},
                )
            elif provider == "Gemini":
                response = client.get(
                    f"https://generativelanguage.googleapis.com/v1beta/models?key={key}",
                )
            else:
                return False, f"Unsupported provider: {provider}"

            if response.status_code in {200, 201}:
                return True, "API key is valid."
            if response.status_code in {401, 403}:
                return False, "API key is not valid. Provider rejected this key."
            if response.status_code == 404 and provider == "Custom OpenAI Compatible Endpoint":
                return True, "API key accepted (endpoint has no /models list)."

            body = (response.text or "")[:200]
            return False, (
                f"API key check failed. Provider returned {response.status_code}. {body}"
            ).strip()
    except httpx.TimeoutException:
        return False, "Could not verify API key (timeout). Check your internet and try again."
    except httpx.RequestError as exc:
        return False, f"Could not verify API key: {exc}"


def validate_upload_file(filename: str, content: bytes, allowed_exts: set[str]) -> tuple[bool, str]:
    """Validate a single uploaded file before indexing."""
    name = (filename or "").strip()
    if not name:
        return False, "File name is empty."

    suffix = Path(name).suffix.lower()
    if suffix not in allowed_exts:
        return False, f"Unsupported type '{suffix}'. Allowed: {', '.join(sorted(allowed_exts))}"
    if not content:
        return False, f"{name} is empty (0 bytes)."
    if len(content) > MAX_UPLOAD_BYTES:
        mb = MAX_UPLOAD_BYTES / (1024 * 1024)
        return False, f"{name} is too large (max {mb:.0f} MB)."

    if suffix in {".txt", ".md", ".csv"}:
        if b"\x00" in content[:2048]:
            return False, f"{name} looks like a binary file, not text."

    return True, "OK"


def credentials_fingerprint(provider: str, api_key: str, model: str, base_url: str = "") -> str:
    """Fingerprint so we invalidate auth when any credential changes."""
    raw = f"{provider}|{(api_key or '').strip()}|{(model or '').strip()}|{(base_url or '').strip()}"
    return re.sub(r"\s+", "", raw)
