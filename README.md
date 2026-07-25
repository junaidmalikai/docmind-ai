<div align="center">

# DocMind AI

**Chat with your documents — a production-grade RAG assistant built entirely on official LangChain v1 + LangGraph components, with a Streamlit UI.**

[![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-UI-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![LangChain](https://img.shields.io/badge/LangChain-v1-1C3C3C)](https://python.langchain.com/)
[![LangGraph](https://img.shields.io/badge/LangGraph-orchestration-FF6F00)](https://langchain-ai.github.io/langgraph/)
[![Chroma](https://img.shields.io/badge/Chroma-vector%20store-4B32C3)](https://www.trychroma.com/)

</div>

---

## Overview

DocMind AI turns your documents into a searchable knowledge base. Upload PDF, DOCX, CSV, Excel, TXT or Markdown files and ask natural-language questions; answers stream back token-by-token, grounded in your content and annotated with source citations.

The RAG pipeline is deliberately built from **official LangChain and LangGraph abstractions** — loaders, splitter, embeddings, Chroma vector store, retrievers, prompt templates, SQL-backed message history, and a compiled `StateGraph` — so the codebase reads like an idiomatic LangChain project. Custom code is confined to the application shell (Streamlit UI, configuration, API-key validation, and the SQLite metadata schema).

## Features

| Capability | Powered by |
|------------|-----------|
| Multi-format ingestion (PDF, DOCX, TXT, MD, CSV, XLSX, XLS) | `PyPDFLoader`, `CSVLoader`, `TextLoader`, `UnstructuredMarkdownLoader`, `Docx2txtLoader`, `UnstructuredExcelLoader` |
| Chunking | `RecursiveCharacterTextSplitter` |
| Embeddings (SentenceTransformers default) | `HuggingFaceEmbeddings` / `OpenAIEmbeddings` |
| Vector store & retrieval | `langchain-chroma` `Chroma` + `VectorStoreRetriever` |
| Prompts & memory | `ChatPromptTemplate` + `MessagesPlaceholder` + `SQLChatMessageHistory` |
| Orchestration & streaming | LangGraph `StateGraph` with `stream_mode="messages"` |
| Multi-provider LLMs | `ChatGroq`, `ChatOpenAI`, `ChatAnthropic`, `ChatGoogleGenerativeAI` |
| Guarded access | Automatic API-key validation gates upload & chat |
| Fast cold start | Heavy libraries load lazily + warm up in a background thread |

## Architecture

```
app.py  →  docmind.ui  (Streamlit)
                │
                ▼
        docmind.services.rag_service.RAGService   (dependency-injection facade)
                │
                ▼
        LangGraph StateGraph
          START
            → load_conversation      (SQLChatMessageHistory)
            → [query_expansion]       (optional, LLM)
            → retrieve                (VectorStoreRetriever over Chroma)
            → [rerank]                (optional, similarity score)
            → generate                (ChatPromptTemplate | chat model, streamed)
            → save                    (persist turn + auto title)
          END
```

## Project layout

```
docmind/
  config/        settings, provider defaults, logging
  models/        Pydantic schemas (LLMSettings, RAGResponse, …)
  state/         LangGraph typed state
  callbacks/     streaming callback handler
  utils/         helpers + API-key validation
  services/
    loaders/       official document loaders + splitter
    embeddings/    HuggingFace / OpenAI embeddings factory
    vectorstore/   langchain-chroma access
    retrieval/     VectorStoreRetriever + optional expansion/rerank
    prompts/       ChatPromptTemplate builders
    llm/           per-provider chat model factory
    memory/        SQLChatMessageHistory adapter
    persistence/   SQLite metadata store (documents + sessions)
    graph/         LangGraph pipeline
    rag_service.py orchestration facade
  ui/            Streamlit theme, sidebar, chat, entry point
app.py           thin `streamlit run` entry
```

## Quick start

Requires Python 3.11/3.12, [uv](https://docs.astral.sh/uv/), and an API key for at least one provider.

```powershell
uv sync
copy .env.example .env
```

Set at least one provider key in `.env`, e.g.:

```dotenv
DEFAULT_PROVIDER=Groq
GROQ_API_KEY=your_key_here
```

Run:

```powershell
uv run streamlit run app.py
```

Open the URL Streamlit prints (usually `http://localhost:8501`).

## Usage

1. Pick a provider in the sidebar and paste your API key — it is validated automatically.
2. Upload one or more documents; each is loaded, split, embedded and indexed into Chroma.
3. Ask questions in the chat panel and watch the streamed, source-cited answer.
4. Revisit, rename or delete earlier conversations from the sidebar.

## Configuration

All settings come from `.env` (see `.env.example`). Highlights:

| Variable | Description | Default |
|----------|-------------|---------|
| `DEFAULT_PROVIDER` | `Groq` / `OpenAI` / `Claude` / `Gemini` / `Custom OpenAI Compatible Endpoint` | `Groq` |
| `EMBEDDING_PROVIDER` | `SentenceTransformers` (default) or `OpenAI` | `SentenceTransformers` |
| `EMBEDDING_MODEL` | Embedding model name | `all-MiniLM-L6-v2` |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | Splitter configuration | `1000` / `150` |
| `MAX_RESULTS` | Chunks retrieved per query | `5` |
| `MAX_HISTORY` | Conversation turns kept in memory | `6` |
| `ENABLE_QUERY_EXPANSION` / `ENABLE_RERANK` | Optional LangGraph stages | `false` / `false` |
| `CHROMA_PATH` / `SQLITE_PATH` / `UPLOAD_DIR` / `CACHE_DIR` | Storage locations | see `.env.example` |

## Performance notes

`torch`, `langchain_core`, `langchain-chroma` and the embedding model are imported on first use, so the UI renders in ~2 seconds. A background thread then preloads the embedding model, vector store and compiled graph so the first question doesn't pay the full load cost.

## Author

**Muhammad Junaid** · [junaidfazal08@gmail.com](mailto:junaidfazal08@gmail.com) · 0304 1659294
