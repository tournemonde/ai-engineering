"""Session 8 persistence and retrieval service."""

from __future__ import annotations

import time
from dataclasses import dataclass

from sqlalchemy import Select, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.generation.rag.chunking.structural import JSONStructuralChunker
from app.generation.rag.embedding.embedder import OpenAIEmbedder
from app.generation.rag.schemas import IngestRequest, IngestResponse, SearchResponse, SearchResult
from app.generation.rag.store.models import ChunkRow, DocumentRow


@dataclass(slots=True)
class DuplicateDocumentError(Exception):
    document_id: int


async def ingest_budget_document(
    *,
    request: IngestRequest,
    session: AsyncSession,
    chunker: JSONStructuralChunker,
    embedder: OpenAIEmbedder,
) -> IngestResponse:
    started = time.perf_counter()

    existing_stmt: Select[tuple[int]] = select(DocumentRow.id).where(
        DocumentRow.source_path == request.source_path
    )
    existing = await session.scalar(existing_stmt)
    if existing is not None:
        raise DuplicateDocumentError(document_id=existing)

    document = DocumentRow(
        source_path=request.source_path,
        document_type=request.document_type,
    )
    session.add(document)
    await session.flush()

    chunks = chunker.chunk([request.content])
    embedded = embedder.embed_many(chunks)

    rows = [
        ChunkRow(
            document_id=document.id,
            chunk_type="budget_component",
            content=chunk.text,
            embedding=chunk.embedding,
            metadata_json=chunk.metadata,
        )
        for chunk in embedded
    ]
    session.add_all(rows)
    await session.commit()

    elapsed_ms = int((time.perf_counter() - started) * 1000)
    embedding_dim = len(embedded[0].embedding) if embedded else 0
    return IngestResponse(
        document_id=document.id,
        chunks_created=len(rows),
        embedding_dimension=embedding_dim,
        ingestion_time_ms=elapsed_ms,
    )


async def semantic_search(
    *,
    query: str,
    k: int,
    session: AsyncSession,
    embedder: OpenAIEmbedder,
) -> SearchResponse:
    started = time.perf_counter()
    query_vector = embedder.embed_one(query)

    stmt = (
        select(
            ChunkRow.id,
            ChunkRow.document_id,
            ChunkRow.chunk_type,
            ChunkRow.content,
            ChunkRow.metadata_json,
            ChunkRow.embedding.cosine_distance(query_vector).label("distance"),
        )
        .where(ChunkRow.embedding.is_not(None))
        .order_by(ChunkRow.embedding.cosine_distance(query_vector))
        .limit(k)
    )
    rows = (await session.execute(stmt)).all()
    elapsed_ms = int((time.perf_counter() - started) * 1000)

    return SearchResponse(
        query=query,
        k=k,
        search_time_ms=elapsed_ms,
        results=[
            SearchResult(
                chunk_id=row.id,
                document_id=row.document_id,
                chunk_type=row.chunk_type,
                content=row.content,
                distance=float(row.distance),
                metadata=row.metadata_json or {},
            )
            for row in rows
        ],
    )
