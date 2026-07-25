"""Central logging configuration."""

from __future__ import annotations

import logging

from docmind.config.settings import settings

_CONFIGURED = False


def configure_logging() -> None:
    """Configure root logging once for the whole process."""
    global _CONFIGURED
    if _CONFIGURED:
        return
    logging.basicConfig(
        level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    )
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    configure_logging()
    return logging.getLogger(name)
