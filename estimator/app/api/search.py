"""Session 8 semantic search endpoint."""

from __future__ import annotations

import structlog
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.dependencies import get_embedder
from app.generation.rag.embedding.embedder import OpenAIEmbedder
from app.generation.rag.schemas import SearchRequest, SearchResponse
from app.generation.rag.store.database import get_async_session
from app.generation.rag.store.service import semantic_search

log = structlog.get_logger()

router = APIRouter(tags=["search"])


@router.post("/search", response_model=SearchResponse)
async def search(
    request: SearchRequest,
    embedder: OpenAIEmbedder | None = Depends(get_embedder),
    session: AsyncSession = Depends(get_async_session),
) -> SearchResponse:
    """Embed the query and return top-k nearest chunks by cosine distance."""
    if embedder is None:
        log.error("embeddings_search_failed", reason="embedder_unavailable")
        raise HTTPException(status_code=500, detail="Embedding service is not available.")

    try:
        return await semantic_search(
            query=request.query,
            k=request.k,
            session=session,
            embedder=embedder,
        )
    except Exception as exc:  # noqa: BLE001
        log.error(
            "embeddings_search_failed",
            reason="search_error",
            error_type=type(exc).__name__,
            error=str(exc)[:300],
        )
        raise HTTPException(status_code=500, detail="Failed to run semantic search.") from exc
