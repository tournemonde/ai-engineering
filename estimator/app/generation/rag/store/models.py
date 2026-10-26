"""ORM models for Session 8 vector persistence."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pgvector.sqlalchemy import Vector
from sqlalchemy import BigInteger, DateTime, ForeignKey, Index, String, Text, func, text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.foundation.persistence.models import Base


class DocumentRow(Base):
    __tablename__ = "documents"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    source_path: Mapped[str] = mapped_column(Text, nullable=False, unique=True)
    document_type: Mapped[str] = mapped_column(String(50), nullable=False)
    ingested_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )
    metadata_json: Mapped[dict[str, Any]] = mapped_column(
        "metadata",
        JSONB,
        nullable=False,
        server_default=text("'{}'::jsonb"),
    )

    chunks: Mapped[list["ChunkRow"]] = relationship(
        back_populates="document",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )


class ChunkRow(Base):
    __tablename__ = "chunks"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    document_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
    )
    chunk_type: Mapped[str] = mapped_column(String(50), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    embedding: Mapped[list[float] | None] = mapped_column(Vector(1536), nullable=True)
    metadata_json: Mapped[dict[str, Any]] = mapped_column(
        "metadata",
        JSONB,
        nullable=False,
        server_default=text("'{}'::jsonb"),
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )

    document: Mapped[DocumentRow] = relationship(back_populates="chunks")


Index("ix_documents_source_path", DocumentRow.source_path)
Index("ix_chunks_document_id", ChunkRow.document_id)
Index("ix_chunks_chunk_type", ChunkRow.chunk_type)
Index("ix_chunks_metadata_gin", ChunkRow.metadata_json, postgresql_using="gin")
