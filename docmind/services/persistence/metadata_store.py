"""SQLite metadata store for indexed documents and chat sessions.

The SQLite *schema* is explicitly allowed to stay custom. Chat *messages* live in
LangChain's ``SQLChatMessageHistory`` table; this store only tracks document
metadata and lightweight session records (title, provider, timestamps).
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    DateTime,
    Integer,
    String,
    create_engine,
    delete,
    func,
    select,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column

from docmind.models import ChatSessionSummary, DocumentInfo
from docmind.utils.helpers import utc_now


class Base(DeclarativeBase):
    pass


class DocumentRow(Base):
    __tablename__ = "docmind_documents"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    filename: Mapped[str] = mapped_column(String, nullable=False)
    file_type: Mapped[str] = mapped_column(String, nullable=False)
    source_path: Mapped[str] = mapped_column(String, nullable=False, default="")
    chunk_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)


class SessionRow(Base):
    __tablename__ = "docmind_sessions"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    title: Mapped[str] = mapped_column(String, default="New Chat")
    provider: Mapped[str] = mapped_column(String, default="Groq")
    model: Mapped[str] = mapped_column(String, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)


class MetadataStore:
    """CRUD for document + session metadata."""

    def __init__(self, sqlite_url: str):
        self.engine = create_engine(sqlite_url, future=True)
        Base.metadata.create_all(self.engine)

    # -- Documents ------------------------------------------------------
    def upsert_document(self, document: DocumentInfo) -> None:
        with Session(self.engine) as session:
            row = session.get(DocumentRow, document.id)
            if row is None:
                row = DocumentRow(id=document.id)
                session.add(row)
            row.filename = document.filename
            row.file_type = document.file_type
            row.source_path = document.source_path
            row.chunk_count = document.chunk_count
            row.created_at = document.created_at or utc_now()
            session.commit()

    def delete_document(self, document_id: str) -> None:
        with Session(self.engine) as session:
            session.execute(delete(DocumentRow).where(DocumentRow.id == document_id))
            session.commit()

    def list_documents(self) -> list[DocumentInfo]:
        with Session(self.engine) as session:
            rows = session.execute(
                select(DocumentRow).order_by(DocumentRow.created_at.desc())
            ).scalars().all()
            return [self._to_document(row) for row in rows]

    def analytics(self) -> dict:
        with Session(self.engine) as session:
            total_documents = session.execute(select(func.count(DocumentRow.id))).scalar_one()
            total_chunks = session.execute(select(func.coalesce(func.sum(DocumentRow.chunk_count), 0))).scalar_one()
        return {"total_documents": int(total_documents), "total_chunks": int(total_chunks)}

    # -- Sessions -------------------------------------------------------
    def create_session(self, session_id: str, provider: str, model: str, title: str = "New Chat") -> None:
        with Session(self.engine) as session:
            if session.get(SessionRow, session_id) is not None:
                return
            now = utc_now()
            session.add(
                SessionRow(
                    id=session_id,
                    title=title,
                    provider=provider,
                    model=model,
                    created_at=now,
                    updated_at=now,
                )
            )
            session.commit()

    def session_title(self, session_id: str) -> str | None:
        with Session(self.engine) as session:
            row = session.get(SessionRow, session_id)
            return row.title if row else None

    def update_session(
        self,
        session_id: str,
        *,
        title: str | None = None,
        provider: str | None = None,
        model: str | None = None,
    ) -> None:
        with Session(self.engine) as session:
            row = session.get(SessionRow, session_id)
            if row is None:
                return
            if title is not None:
                row.title = title
            if provider is not None:
                row.provider = provider
            if model is not None:
                row.model = model
            row.updated_at = utc_now()
            session.commit()

    def touch_session(self, session_id: str) -> None:
        self.update_session(session_id)

    def delete_session(self, session_id: str) -> None:
        with Session(self.engine) as session:
            session.execute(delete(SessionRow).where(SessionRow.id == session_id))
            session.commit()

    def list_sessions(self, search: str = "") -> list[ChatSessionSummary]:
        with Session(self.engine) as session:
            stmt = select(SessionRow).order_by(SessionRow.updated_at.desc())
            term = (search or "").strip()
            if term:
                stmt = stmt.where(SessionRow.title.ilike(f"%{term}%"))
            rows = session.execute(stmt).scalars().all()
            return [
                ChatSessionSummary(
                    session_id=row.id,
                    title=row.title,
                    updated_at=row.updated_at,
                    created_at=row.created_at,
                    provider=row.provider,
                    model=row.model,
                )
                for row in rows
            ]

    @staticmethod
    def _to_document(row: DocumentRow) -> DocumentInfo:
        return DocumentInfo(
            id=row.id,
            filename=row.filename,
            file_type=row.file_type,
            source_path=row.source_path,
            chunk_count=row.chunk_count,
            created_at=row.created_at,
        )
