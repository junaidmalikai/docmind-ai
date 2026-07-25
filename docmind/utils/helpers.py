"""Small, dependency-free helpers."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def safe_filename(name: str) -> str:
    return Path(name).name


def ensure_dir(path: str | Path) -> Path:
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    return target
