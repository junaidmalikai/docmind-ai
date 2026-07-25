"""Prompt templates built with official ChatPromptTemplate APIs."""

from docmind.services.prompts.templates import (
    build_general_prompt,
    build_query_expansion_prompt,
    build_rag_prompt,
    build_title_prompt,
)

__all__ = [
    "build_rag_prompt",
    "build_general_prompt",
    "build_title_prompt",
    "build_query_expansion_prompt",
]
