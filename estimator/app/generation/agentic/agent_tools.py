"""Flat Responses-API tool schemas + deterministic implementations (Session 12).

``search_budgets`` wraps the real S9/S10 ``retrieve()`` (injectable backend for
offline / ``--stub`` debugging). ``calculate_estimate`` and ``validate_estimate``
are pure Python — no LLM.
"""

from __future__ import annotations

import json
import statistics
from collections.abc import Awaitable, Callable
from typing import Any

import structlog

from app.config import get_settings

log = structlog.get_logger()

# Flat contingency buffer — transparent, no hidden multipliers.
CONTINGENCY_FACTOR = 0.15

# Hours outside this band on a single component trip a validation warning.
_MIN_REASONABLE_HOURS = 8.0
_MAX_REASONABLE_HOURS = 4000.0

RetrievalBackend = Callable[[str, dict[str, Any] | None], Awaitable[list[dict[str, Any]]]]


# --- Responses API tool schemas (flat, strict:true) --------------------------

TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "name": "search_budgets",
        "description": (
            "Search historical project budgets for items comparable to a SINGLE "
            "software component. Call once per component; never combine unrelated "
            "components (e.g. an ERP integration and a mobile app) in one query — "
            "mixed results cannot be compared. Returns comparable items with "
            "estimated_hours, budget_id, sector and distance."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Focused description of one component to price.",
                },
                "filters": {
                    "type": ["object", "null"],
                    "description": "Optional filters; pass null when unused.",
                    "properties": {
                        "component_type": {
                            "type": ["string", "null"],
                            "enum": [
                                "integration",
                                "migration",
                                "frontend",
                                "backend",
                                "mobile",
                                "analytics",
                                None,
                            ],
                        },
                        "sectors": {
                            "type": ["array", "null"],
                            "items": {"type": "string"},
                        },
                    },
                    "required": ["component_type", "sectors"],
                    "additionalProperties": False,
                },
            },
            "required": ["query", "filters"],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "type": "function",
        "name": "calculate_estimate",
        "description": (
            "Deterministically cost each component from its historical reference "
            "amounts (median + contingency), then sum. Call AFTER you have gathered "
            "references via search_budgets for every component. Do NOT invent amounts."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "components": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "reference_amounts": {
                                "type": "array",
                                "items": {"type": "number"},
                            },
                        },
                        "required": ["name", "reference_amounts"],
                        "additionalProperties": False,
                    },
                },
            },
            "required": ["components"],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "type": "function",
        "name": "validate_estimate",
        "description": (
            "Run deterministic guardrails on a candidate estimate: empty references, "
            "unreasonable per-component hours, and total mismatch. Call as the LAST "
            "tool step before producing the final structured answer. Returns issues[] "
            "— empty means ok."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "components": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "reference_amounts": {
                                "type": "array",
                                "items": {"type": "number"},
                            },
                            "estimated_hours": {"type": "number"},
                        },
                        "required": ["name", "reference_amounts", "estimated_hours"],
                        "additionalProperties": False,
                    },
                },
                "total_hours": {"type": "number"},
            },
            "required": ["components", "total_hours"],
            "additionalProperties": False,
        },
        "strict": True,
    },
]


# --- Implementations ---------------------------------------------------------


def calculate_estimate(args: dict[str, Any]) -> dict[str, Any]:
    """Cost each component from reference amounts (median + contingency)."""
    components = args["components"]
    breakdown: list[dict[str, Any]] = []
    total = 0.0

    for component in components:
        name = component["name"]
        refs = list(component.get("reference_amounts") or [])
        if refs:
            central = float(statistics.median(refs))
            hours = round(central * (1 + CONTINGENCY_FACTOR), 1)
            unbudgeted = False
        else:
            hours = 0.0
            unbudgeted = True

        total += hours
        breakdown.append(
            {
                "name": name,
                "reference_count": len(refs),
                "estimated_hours": hours,
                "unbudgeted": unbudgeted,
            }
        )

    total = round(total, 1)
    return {
        "components": breakdown,
        "total_hours": total,
        "summary": f"total={total}h across {len(breakdown)} components",
    }


def validate_estimate(args: dict[str, Any]) -> dict[str, Any]:
    """Deterministic guardrails over a candidate estimate."""
    components = args["components"]
    total_hours = float(args["total_hours"])
    issues: list[str] = []
    summed = 0.0

    for component in components:
        name = component["name"]
        refs = list(component.get("reference_amounts") or [])
        hours = float(component.get("estimated_hours") or 0.0)
        summed += hours
        if not refs:
            issues.append(f"{name}: no reference amounts (unbudgeted)")
        if hours < _MIN_REASONABLE_HOURS and refs:
            issues.append(f"{name}: estimated_hours={hours} below {_MIN_REASONABLE_HOURS}")
        if hours > _MAX_REASONABLE_HOURS:
            issues.append(f"{name}: estimated_hours={hours} above {_MAX_REASONABLE_HOURS}")

    summed = round(summed, 1)
    if abs(summed - total_hours) > 0.5:
        issues.append(f"total_hours={total_hours} != sum(components)={summed}")

    return {
        "ok": not issues,
        "issues": issues,
        "summary": "ok" if not issues else f"{len(issues)} issue(s): " + "; ".join(issues),
    }


async def search_budgets_via_retrieve(
    query: str,
    filters: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Wrap S9/S10 retrieve() filtered to historical_task chunks."""
    import asyncio

    from app.dependencies import get_embedder, get_runtime_retrieval_config
    from app.generation.rag.retrieval.collections import Collection
    from app.generation.rag.retrieval.pipeline import retrieve

    settings = get_settings()
    embedder = get_embedder()
    if embedder is None:
        raise RuntimeError("Embedding service is not available (no OpenAI key).")

    runtime = get_runtime_retrieval_config()
    embedding = await asyncio.to_thread(embedder.embed_one, query)

    sectors = None
    if filters and filters.get("sectors"):
        sectors = list(filters["sectors"])

    result = await retrieve(
        query_embedding=embedding,
        query_text=query,
        search_mode=runtime.search_mode,
        rerank=runtime.reranker_enabled,
        top_k=settings.AGENT_SEARCH_TOP_K,
        recall_k=settings.RETRIEVAL_RECALL_TOP_K,
        rerank_top_n=settings.AGENT_SEARCH_TOP_K,
        distance_threshold=settings.AGENT_SEARCH_DISTANCE_THRESHOLD,
        rrf_k=settings.RRF_K,
        collection=Collection.BUDGET,
        sectors=sectors,
        chunk_types=["historical_task"],
    )

    hits: list[dict[str, Any]] = []
    for chunk in result.chunks:
        hits.append(
            {
                "id": chunk.id,
                "content_preview": (chunk.content or "")[:240],
                "sector": chunk.sector,
                "budget_id": chunk.budget_id,
                "estimated_hours": chunk.estimated_hours,
                "distance": chunk.distance,
            }
        )
    return hits


async def dispatch_tool(
    name: str,
    args: dict[str, Any],
    *,
    retrieval_backend: RetrievalBackend | None = None,
) -> str:
    """Run a tool and return a JSON string (observation for the model).

    Tool errors become an observation envelope so the model can self-correct
    instead of crashing the loop.
    """
    try:
        if name == "search_budgets":
            backend = retrieval_backend or search_budgets_via_retrieve
            filters = args.get("filters")
            hits = await backend(args["query"], filters)
            # Sector / type filters are often guessed wrong by the model; if they
            # wipe the result set, retry once unfiltered so the loop can still price.
            if not hits and filters:
                hits = await backend(args["query"], None)
                retried = True
            else:
                retried = False
            payload: dict[str, Any] = {
                "count": len(hits),
                "items": hits,
                "filters_relaxed": retried,
                "summary": (
                    f"{len(hits)} hit(s)"
                    + (" (filters relaxed)" if retried and hits else "")
                    if hits
                    else "no matches — reformulate or mark component unbudgeted"
                ),
            }
        elif name == "calculate_estimate":
            payload = calculate_estimate(args)
        elif name == "validate_estimate":
            payload = validate_estimate(args)
        else:
            payload = {"error": f"unknown tool: {name}"}
    except Exception as exc:  # noqa: BLE001 — surface to the model as observation
        log.warning("agent_tool_failed", tool=name, error=str(exc)[:200])
        payload = {"error": f"{type(exc).__name__}: {exc}"}

    return json.dumps(payload, ensure_ascii=False, default=str)
