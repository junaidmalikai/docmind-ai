"""DocMind AI — a production-grade RAG assistant built on LangChain + LangGraph.

The package is organized as thin, single-responsibility services that wrap
*official* LangChain / LangGraph abstractions. Business logic (Streamlit UI,
configuration, API-key validation, SQLite schema) stays custom; everything RAG
(loading, splitting, embeddings, vector store, retrieval, prompts, memory,
orchestration, streaming) is delegated to LangChain.
"""

__all__ = ["__version__"]

__version__ = "1.0.0"
