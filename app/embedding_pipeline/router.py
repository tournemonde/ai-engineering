"""HTTP layer for the embedding pipeline."""

from __future__ import annotations

import structlog
from fastapi import APIRouter, Depends, HTTPException

from app.dependencies import get_chunker, get_embedder
from app.embedding_pipeline.chunker import JSONStructuralChunker
from app.embedding_pipeline.embedder import OpenAIEmbedder, estimated_cost_usd
from app.embedding_pipeline.schemas import IngestRequest, IngestResponse, IngestStats

log = structlog.get_logger()

router = APIRouter(prefix="/embeddings", tags=["embeddings"])


@router.post("/ingest", response_model=IngestResponse)
def ingest(
    request: IngestRequest,
    chunker: JSONStructuralChunker = Depends(get_chunker),
    embedder: OpenAIEmbedder | None = Depends(get_embedder),
) -> IngestResponse:
    """Chunk budgets, embed every chunk, and return vectors plus aggregate stats."""
    if embedder is None:
        log.error("embeddings_ingest_failed", reason="embedder_unavailable")
        raise HTTPException(
            status_code=500, detail="Embedding service is not available."
        )

    chunks = chunker.chunk(request.budgets)
    log.info(
        "embeddings_ingest_received",
        total_budgets=len(request.budgets),
        total_chunks=len(chunks),
    )

    try:
        embedded = embedder.embed_many(chunks)
    except Exception as exc:  # noqa: BLE001
        log.error(
            "embeddings_ingest_failed",
            reason="embedding_api_error",
            error_type=type(exc).__name__,
            error=str(exc)[:400],
        )
        raise HTTPException(
            status_code=500,
            detail="Embedding service is not available.",
        ) from exc

    total_tokens = sum(chunk.token_count for chunk in chunks)
    stats = IngestStats(
        total_budgets=len(request.budgets),
        total_chunks=len(chunks),
        total_tokens=total_tokens,
        estimated_cost_usd=estimated_cost_usd(total_tokens),
    )
    return IngestResponse(chunks=embedded, stats=stats)
