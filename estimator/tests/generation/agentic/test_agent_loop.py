"""Network-free test of the Session 12 agent loop with a scripted fake client."""

from __future__ import annotations

import json
from typing import Any

import pytest

from app.generation.agentic.agent_loop import run_estimation_agent
from app.generation.agentic.agent_schemas import AgentEstimate, AgentTrace


class _FakeFunctionCall:
    type = "function_call"

    def __init__(self, name: str, arguments: dict[str, Any], call_id: str) -> None:
        self.name = name
        self.arguments = json.dumps(arguments)
        self.call_id = call_id


class _FakeResponse:
    def __init__(
        self,
        *,
        response_id: str,
        output: list[Any],
        output_parsed: AgentEstimate | None = None,
    ) -> None:
        self.id = response_id
        self.output = output
        self.output_parsed = output_parsed


class _FakeResponsesAPI:
    def __init__(self, script: list[_FakeResponse]) -> None:
        self._script = list(script)
        self.calls: list[dict[str, Any]] = []

    async def parse(self, **kwargs: Any) -> _FakeResponse:
        self.calls.append(kwargs)
        if not self._script:
            raise AssertionError("Unexpected extra responses.parse call")
        return self._script.pop(0)


class _FakeAsyncOpenAI:
    def __init__(self, script: list[_FakeResponse]) -> None:
        self.responses = _FakeResponsesAPI(script)


@pytest.mark.asyncio
async def test_agent_loop_searches_calculates_validates_and_finishes() -> None:
    estimate = AgentEstimate(
        components=[
            {
                "name": "Auth",
                "hours": 115.0,
                "reference_budget_ids": ["B1"],
                "unbudgeted": False,
            }
        ],
        total_hours=115.0,
        notes="ok",
        confidence="high",
    )

    script = [
        _FakeResponse(
            response_id="r1",
            output=[
                _FakeFunctionCall(
                    "search_budgets",
                    {"query": "oauth auth", "filters": None},
                    "call_search",
                )
            ],
        ),
        _FakeResponse(
            response_id="r2",
            output=[
                _FakeFunctionCall(
                    "calculate_estimate",
                    {
                        "components": [
                            {"name": "Auth", "reference_amounts": [100.0]}
                        ]
                    },
                    "call_calc",
                )
            ],
        ),
        _FakeResponse(
            response_id="r3",
            output=[
                _FakeFunctionCall(
                    "validate_estimate",
                    {
                        "components": [
                            {
                                "name": "Auth",
                                "reference_amounts": [100.0],
                                "estimated_hours": 115.0,
                            }
                        ],
                        "total_hours": 115.0,
                    },
                    "call_val",
                )
            ],
        ),
        _FakeResponse(
            response_id="r4",
            output=[],
            output_parsed=estimate,
        ),
    ]

    async def backend(query: str, filters: dict | None = None):
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

    client = _FakeAsyncOpenAI(script)
    result, trace = await run_estimation_agent(
        "Need an OAuth auth backend.",
        client=client,  # type: ignore[arg-type]
        model="gpt-5-mini",
        effort="minimal",
        max_iterations=8,
        retrieval_backend=backend,
    )

    assert isinstance(result, AgentEstimate)
    assert result.total_hours == 115.0
    assert isinstance(trace, AgentTrace)
    assert len(trace.steps) == 3
    assert "search_budgets" in trace.steps[0].action
    assert "calculate_estimate" in trace.steps[1].action
    assert "validate_estimate" in trace.steps[2].action
    rendered = trace.render()
    assert "STEP 1" in rendered
    assert "reasoning:" in rendered
    assert len(client.responses.calls) == 4
    # Continuations must chain previous_response_id and only send tool outputs.
    assert client.responses.calls[1]["previous_response_id"] == "r1"
    assert client.responses.calls[1]["input"][0]["type"] == "function_call_output"
