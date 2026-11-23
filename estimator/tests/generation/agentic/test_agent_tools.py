"""Network-free tests for Session 12 deterministic tools."""

from __future__ import annotations

import json

import pytest

from app.generation.agentic.agent_tools import (
    CONTINGENCY_FACTOR,
    calculate_estimate,
    dispatch_tool,
    validate_estimate,
)


def test_calculate_estimate_median_plus_contingency() -> None:
    result = calculate_estimate(
        {
            "components": [
                {"name": "Auth", "reference_amounts": [100.0, 200.0, 300.0]},
                {"name": "Mobile", "reference_amounts": []},
            ]
        }
    )
    # median(100,200,300)=200 → 200 * 1.15 = 230
    assert result["components"][0]["estimated_hours"] == round(200 * (1 + CONTINGENCY_FACTOR), 1)
    assert result["components"][0]["unbudgeted"] is False
    assert result["components"][1]["estimated_hours"] == 0.0
    assert result["components"][1]["unbudgeted"] is True
    assert result["total_hours"] == result["components"][0]["estimated_hours"]
    assert "total=" in result["summary"]


def test_validate_estimate_flags_empty_refs_and_total_mismatch() -> None:
    result = validate_estimate(
        {
            "components": [
                {
                    "name": "Auth",
                    "reference_amounts": [],
                    "estimated_hours": 0.0,
                },
                {
                    "name": "Core",
                    "reference_amounts": [100.0],
                    "estimated_hours": 115.0,
                },
            ],
            "total_hours": 999.0,
        }
    )
    assert result["ok"] is False
    assert any("unbudgeted" in i for i in result["issues"])
    assert any("total_hours" in i for i in result["issues"])


def test_validate_estimate_ok() -> None:
    result = validate_estimate(
        {
            "components": [
                {
                    "name": "Auth",
                    "reference_amounts": [100.0],
                    "estimated_hours": 115.0,
                }
            ],
            "total_hours": 115.0,
        }
    )
    assert result["ok"] is True
    assert result["issues"] == []


@pytest.mark.asyncio
async def test_dispatch_unknown_tool_returns_error_envelope() -> None:
    raw = await dispatch_tool("nope", {})
    data = json.loads(raw)
    assert "error" in data


@pytest.mark.asyncio
async def test_dispatch_search_uses_injected_backend() -> None:
    async def backend(query: str, filters: dict | None = None):
        assert "auth" in query.lower()
        return [
            {
                "id": 1,
                "content_preview": "oauth",
                "sector": "finance",
                "budget_id": "B1",
                "estimated_hours": 100.0,
                "distance": 0.1,
            }
        ]

    raw = await dispatch_tool(
        "search_budgets",
        {"query": "oauth auth backend", "filters": None},
        retrieval_backend=backend,
    )
    data = json.loads(raw)
    assert data["count"] == 1
    assert data["items"][0]["budget_id"] == "B1"
