"""Load and split documents with official LangChain loaders + splitter.

No custom parsing lives here: each file type is delegated to an official
``langchain_community`` loader, and chunking is delegated to
``RecursiveCharacterTextSplitter``. This module only wires those together and
attaches stable identifiers/citation metadata used for retrieval and deletion.
"""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from typing import TYPE_CHECKING, Callable

from docmind.config.logging import get_logger

if TYPE_CHECKING:
    from langchain_core.documents import Document

logger = get_logger(__name__)

SUPPORTED_EXTENSIONS: set[str] = {".pdf", ".csv", ".xlsx", ".xls", ".docx", ".txt", ".md"}


def _build_loader(path: str, suffix: str):
    """Return the official LangChain loader for a file extension."""
    from langchain_community.document_loaders import (
        CSVLoader,
        Docx2txtLoader,
        PyPDFLoader,
        TextLoader,
        UnstructuredExcelLoader,
        UnstructuredMarkdownLoader,
    )

    loaders: dict[str, Callable[[], object]] = {
        ".pdf": lambda: PyPDFLoader(path),
        ".csv": lambda: CSVLoader(path, encoding="utf-8"),
        ".txt": lambda: TextLoader(path, encoding="utf-8", autodetect_encoding=True),
        ".md": lambda: UnstructuredMarkdownLoader(path),
        ".docx": lambda: Docx2txtLoader(path),
        ".xlsx": lambda: UnstructuredExcelLoader(path, mode="elements"),
        ".xls": lambda: UnstructuredExcelLoader(path, mode="elements"),
    }
    if suffix not in loaders:
        raise ValueError(
            f"Unsupported file type '{suffix}'. "
            f"Allowed: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )
    return loaders[suffix]()


def _citation(filename: str, metadata: dict) -> str:
    """Human-readable source label derived from official loader metadata."""
    parts = [filename]
    page = metadata.get("page")
    if page is not None:
        try:
            parts.append(f"page {int(page) + 1}")  # PyPDFLoader pages are 0-based
        except (TypeError, ValueError):
            parts.append(f"page {page}")
    sheet = metadata.get("page_name")
    if sheet:
        parts.append(f"sheet {sheet}")
    row = metadata.get("row")
    if row is not None:
        parts.append(f"row {row}")
    return ", ".join(str(p) for p in parts)


class DocumentLoaderService:
    """Loads a file into LangChain ``Document`` chunks ready for a vector store."""

    def __init__(self, chunk_size: int, chunk_overlap: int):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    @staticmethod
    def document_id(path: Path, content: bytes | None = None) -> str:
        raw = content if content is not None else path.read_bytes()
        return f"{path.stem}-{sha256(raw).hexdigest()[:16]}"

    def load_and_split(self, file_path: str) -> tuple[str, list["Document"]]:
        """Return ``(document_id, chunks)`` for a file on disk."""
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        path = Path(file_path)
        suffix = path.suffix.lower()
        doc_id = self.document_id(path)

        loader = _build_loader(str(path), suffix)
        raw_docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            keep_separator=True,
            add_start_index=True,
        )
        chunks = splitter.split_documents(raw_docs)

        filename = path.name
        file_type = suffix.lstrip(".")
        cleaned: list["Document"] = []
        for index, chunk in enumerate(chunks):
            if not (chunk.page_content or "").strip():
                continue
            source_meta = chunk.metadata or {}
            chunk.metadata = {
                "document_id": doc_id,
                "chunk_id": f"{doc_id}-{index}",
                "chunk_index": index,
                "filename": filename,
                "file_type": file_type,
                "citation": _citation(filename, source_meta),
            }
            page = source_meta.get("page")
            if page is not None:
                try:
                    chunk.metadata["page_number"] = int(page) + 1
                except (TypeError, ValueError):
                    pass
            if source_meta.get("page_name"):
                chunk.metadata["sheet_name"] = str(source_meta["page_name"])
            if source_meta.get("row") is not None:
                try:
                    chunk.metadata["row_number"] = int(source_meta["row"])
                except (TypeError, ValueError):
                    pass
            cleaned.append(chunk)

        logger.info("Loaded %s → %d chunks", filename, len(cleaned))
        return doc_id, cleaned
