"""HTTP layer for the embedding pipeline.

Thin router: it orchestrates chunker -> embedder -> response assembly and maps
failures to status codes. No business logic lives here.
"""

from __future__ import annotations

import structlog
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.dependencies import ALL_STRATEGIES, build_chunkers, get_chunker, get_embedder
from app.generation.rag.chunking.structural import JSONStructuralChunker
from app.generation.rag.analysis.comparison import (
    ChunkingComparator,
    CompareRequest,
    CompareResponse,
)
from app.generation.rag.embedding.embedder import OpenAIEmbedder
from app.generation.rag.schemas import (
    IngestRequest,
    IngestResponse,
)
from app.generation.rag.store.database import get_async_session
from app.generation.rag.store.service import (
    DuplicateDocumentError,
    ingest_budget_document,
)

log = structlog.get_logger()

router = APIRouter(prefix="/embeddings", tags=["embeddings"])


@router.post("/ingest", response_model=IngestResponse)
async def ingest(
    request: IngestRequest,
    chunker: JSONStructuralChunker = Depends(get_chunker),
    embedder: OpenAIEmbedder | None = Depends(get_embedder),
    session: AsyncSession = Depends(get_async_session),
) -> IngestResponse:
    """Persist a budget document and its embedded component chunks."""
    if embedder is None:
        # No OPENAI_API_KEY configured. Generic message to the client, detail logged.
        log.error("embeddings_ingest_failed", reason="embedder_unavailable")
        raise HTTPException(status_code=500, detail="Embedding service is not available.")

    try:
        response = await ingest_budget_document(
            request=request,
            session=session,
            chunker=chunker,
            embedder=embedder,
        )
        log.info(
            "embeddings_ingest_done",
            document_id=response.document_id,
            chunks_created=response.chunks_created,
            embedding_dimension=response.embedding_dimension,
            ingestion_time_ms=response.ingestion_time_ms,
        )
        return response
    except DuplicateDocumentError as exc:
        raise HTTPException(
            status_code=409,
            detail={"detail": "Document already ingested", "document_id": exc.document_id},
        ) from exc
    except Exception as exc:  # noqa: BLE001 — any embedding-API failure becomes a 500.
        log.error(
            "embeddings_ingest_failed",
            reason="embedding_api_error",
            error_type=type(exc).__name__,
            error=str(exc)[:300],
        )
        raise HTTPException(status_code=500, detail="Failed to persist embeddings.") from exc




@router.post("/compare", response_model=CompareResponse)
def compare(
    request: CompareRequest,
    embedder: OpenAIEmbedder | None = Depends(get_embedder),
) -> CompareResponse:
    """Run several chunking strategies over the same budgets and compare them.

    Returns per-strategy corpus stats and, if queries are given, the top-k
    chunks each strategy retrieves. Nothing is persisted (Session 8 territory).
    """
    if embedder is None:
        log.error("embeddings_compare_failed", reason="embedder_unavailable")
        raise HTTPException(status_code=500, detail="Embedding service is not available.")

    names = request.strategies or ALL_STRATEGIES
    try:
        chunkers = build_chunkers(names)
    except KeyError as exc:
        raise HTTPException(status_code=400, detail=f"Unknown strategy: {exc.args[0]}") from exc
    except RuntimeError as exc:
        # A strategy needs an API key that is not configured.
        log.error("embeddings_compare_failed", reason="missing_api_key", error=str(exc))
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    comparator = ChunkingComparator(chunkers, embedder)
    log.info(
        "embeddings_compare_received",
        total_budgets=len(request.budgets),
        strategies=names,
        n_queries=len(request.queries),
    )
    try:
        stats = comparator.compute_stats(request.budgets)
        queries = comparator.run_queries(request.budgets, request.queries, request.top_k)
    except Exception as exc:  # noqa: BLE001 — any chunker/embedding failure becomes a 500.
        log.error(
            "embeddings_compare_failed",
            reason="comparison_error",
            error_type=type(exc).__name__,
            error=str(exc)[:300],
        )
        raise HTTPException(status_code=500, detail="Failed to run chunking comparison.") from exc

    return CompareResponse(stats_per_strategy=stats, queries_per_strategy=queries)
