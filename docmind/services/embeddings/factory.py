"""Factory returning an official LangChain ``Embeddings`` implementation.

SentenceTransformers (via ``langchain_huggingface.HuggingFaceEmbeddings``) is the
default backend, with OpenAI / HuggingFace-API alternatives available through
configuration. No embeddings are computed by hand.
"""

from __future__ import annotations

import os
from functools import lru_cache

from docmind.config.logging import get_logger
from docmind.config.settings import settings

# XET downloads frequently fail with ConnectionReset on Windows networks.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

logger = get_logger(__name__)


@lru_cache(maxsize=4)
def create_embeddings(
    provider: str | None = None,
    model_name: str | None = None,
):
    """Create (and process-cache) an official LangChain Embeddings instance."""
    provider = (provider or settings.EMBEDDING_PROVIDER or "SentenceTransformers").strip()
    model_name = (model_name or settings.EMBEDDING_MODEL).strip()
    key = provider.lower()

    if key in {"openai", "openaiembeddings"}:
        from langchain_openai import OpenAIEmbeddings

        logger.info("Using OpenAIEmbeddings (%s)", model_name)
        return OpenAIEmbeddings(model=model_name or "text-embedding-3-small")

    # Default: SentenceTransformers through the official HuggingFace wrapper.
    from langchain_huggingface import HuggingFaceEmbeddings

    logger.info("Using HuggingFaceEmbeddings/SentenceTransformers (%s)", model_name)
    return HuggingFaceEmbeddings(
        model_name=model_name or "all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
