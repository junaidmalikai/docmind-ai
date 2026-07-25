"""Application configuration and provider defaults."""

from docmind.config.providers import (
    DEFAULT_PROVIDER,
    PROVIDER_DEFAULTS,
    provider_settings,
    resolve_model,
    resolve_provider,
)
from docmind.config.settings import PROJECT_ROOT, Settings, get_settings, settings

__all__ = [
    "PROJECT_ROOT",
    "Settings",
    "get_settings",
    "settings",
    "PROVIDER_DEFAULTS",
    "DEFAULT_PROVIDER",
    "provider_settings",
    "resolve_provider",
    "resolve_model",
]
