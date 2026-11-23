"""Trace rendering for Session 12."""

from __future__ import annotations

from app.generation.agentic.agent_schemas import AgentStep, AgentTrace


def test_agent_trace_render_step_format() -> None:
    trace = AgentTrace(
        steps=[
            AgentStep(
                reasoning="two components",
                action='search_budgets({"query":"ERP"})',
                observation="4 hits",
            ),
            AgentStep(
                reasoning="price them",
                action="calculate_estimate(...)",
                observation="total=410h",
            ),
        ]
    )
    rendered = trace.render()
    assert "STEP 1" in rendered
    assert "STEP 2" in rendered
    assert "reasoning:   two components" in rendered
    assert "action:      search_budgets" in rendered
    assert "observation: 4 hits" in rendered
