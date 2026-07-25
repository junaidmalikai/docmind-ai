"""Chat model construction using official LangChain provider integrations."""

from docmind.services.llm.factory import LLMConfigurationError, create_chat_model

__all__ = ["create_chat_model", "LLMConfigurationError"]
