"""Document loading + splitting using official LangChain components."""

from docmind.services.loaders.document_loader import (
    SUPPORTED_EXTENSIONS,
    DocumentLoaderService,
)

__all__ = ["DocumentLoaderService", "SUPPORTED_EXTENSIONS"]
