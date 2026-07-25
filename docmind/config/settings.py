"""Environment-driven application configuration.

This is intentionally custom (application configuration is explicitly allowed to
stay custom). Everything else in the project consumes these values to configure
official LangChain components.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
# Ambient environment variables take precedence; .env only fills the gaps.
load_dotenv(PROJECT_ROOT / ".env", override=False)


def _path(name: str, default: Path) -> str:
    value = os.getenv(name)
    if not value:
        return str(default)
    path = Path(value)
    return str(path if path.is_absolute() else PROJECT_ROOT / path)


def _int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


def _float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        return default


def _bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    """Central configuration loaded from environment variables."""

    # LLM provider credentials / defaults
    DEFAULT_PROVIDER: str = field(default_factory=lambda: os.getenv("DEFAULT_PROVIDER", "Groq"))
    OPENAI_API_KEY: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    OPENAI_MODEL: str = field(default_factory=lambda: os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
    GROQ_API_KEY: str = field(default_factory=lambda: os.getenv("GROQ_API_KEY", ""))
    GROQ_MODEL: str = field(default_factory=lambda: os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"))
    ANTHROPIC_API_KEY: str = field(default_factory=lambda: os.getenv("ANTHROPIC_API_KEY", ""))
    CLAUDE_MODEL: str = field(default_factory=lambda: os.getenv("CLAUDE_MODEL", "claude-3-5-haiku-20241022"))
    GEMINI_API_KEY: str = field(default_factory=lambda: os.getenv("GEMINI_API_KEY", ""))
    GEMINI_MODEL: str = field(default_factory=lambda: os.getenv("GEMINI_MODEL", "gemini-2.0-flash"))
    CUSTOM_BASE_URL: str = field(default_factory=lambda: os.getenv("CUSTOM_BASE_URL", ""))
    CUSTOM_API_KEY: str = field(default_factory=lambda: os.getenv("CUSTOM_API_KEY", ""))
    CUSTOM_MODEL: str = field(default_factory=lambda: os.getenv("CUSTOM_MODEL", ""))

    # Generation settings
    TEMPERATURE: float = field(default_factory=lambda: _float("TEMPERATURE", 0.7))
    TOP_P: float = field(default_factory=lambda: _float("TOP_P", 0.9))
    MAX_TOKENS: int = field(default_factory=lambda: _int("MAX_TOKENS", 2048))
    STREAMING: bool = field(default_factory=lambda: _bool("STREAMING", True))

    # Embeddings & retrieval
    EMBEDDING_PROVIDER: str = field(default_factory=lambda: os.getenv("EMBEDDING_PROVIDER", "SentenceTransformers"))
    EMBEDDING_MODEL: str = field(default_factory=lambda: os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2"))
    CHUNK_SIZE: int = field(default_factory=lambda: _int("CHUNK_SIZE", 1000))
    CHUNK_OVERLAP: int = field(default_factory=lambda: _int("CHUNK_OVERLAP", 150))
    MAX_RESULTS: int = field(default_factory=lambda: _int("MAX_RESULTS", 5))
    MAX_HISTORY: int = field(default_factory=lambda: _int("MAX_HISTORY", 6))

    # Optional LangGraph stages
    ENABLE_QUERY_EXPANSION: bool = field(default_factory=lambda: _bool("ENABLE_QUERY_EXPANSION", False))
    ENABLE_RERANK: bool = field(default_factory=lambda: _bool("ENABLE_RERANK", False))

    # Storage paths
    CHROMA_PATH: str = field(default_factory=lambda: _path("CHROMA_PATH", PROJECT_ROOT / "data" / "chroma_db"))
    SQLITE_PATH: str = field(default_factory=lambda: _path("SQLITE_PATH", PROJECT_ROOT / "data" / "rag.sqlite3"))
    UPLOAD_DIR: str = field(default_factory=lambda: _path("UPLOAD_DIR", PROJECT_ROOT / "uploads"))
    CACHE_DIR: str = field(default_factory=lambda: _path("CACHE_DIR", PROJECT_ROOT / "cache"))

    # Logging
    LOG_LEVEL: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))

    @property
    def sqlite_url(self) -> str:
        """SQLAlchemy connection URL used by both the metadata store and chat memory."""
        return f"sqlite:///{self.SQLITE_PATH}"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Cached settings singleton (dependency-injection friendly)."""
    return Settings()


settings = get_settings()
