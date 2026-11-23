"""Schemas for the Session 12 hand-written estimation agent.

Trace models (``AgentStep`` / ``AgentTrace``), the LIGHT result shape
(``AgentEstimate`` — deliberately distinct from the heavy RAG ``Estimate``),
and typed tool-argument models used to validate what the model emits.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


# --- Tool argument models ----------------------------------------------------


class SearchBudgetsArgs(BaseModel):
    query: str = Field(description="Focused description of ONE component to price.")
    filters: dict[str, Any] | None = Field(
        default=None,
        description="Optional filters (e.g. sectors, component_type).",
    )


class EstimateComponentInput(BaseModel):
    name: str
    reference_amounts: list[float] = Field(default_factory=list)


class CalculateEstimateArgs(BaseModel):
    components: list[EstimateComponentInput]


class ValidateEstimateArgs(BaseModel):
    components: list[EstimateComponentInput]
    total_hours: float = Field(ge=0)


# --- Result + trace ----------------------------------------------------------


class AgentComponentEstimate(BaseModel):
    name: str
    hours: float = Field(ge=0)
    reference_budget_ids: list[str] = Field(default_factory=list)
    unbudgeted: bool = False


class AgentEstimate(BaseModel):
    """Structured estimate produced by the agent (hours-based, light)."""

    components: list[AgentComponentEstimate]
    total_hours: float = Field(ge=0)
    notes: str = ""
    confidence: str = Field(
        default="medium",
        description="high | medium | low — agent's self-reported confidence.",
    )


class AgentStep(BaseModel):
    reasoning: str
    action: str
    observation: str


class AgentTrace(BaseModel):
    steps: list[AgentStep] = Field(default_factory=list)

    def render(self) -> str:
        """Console-friendly STEP N format required by the exercise."""
        lines: list[str] = []
        for i, step in enumerate(self.steps, start=1):
            lines.append(f"STEP {i}")
            lines.append(f"  reasoning:   {step.reasoning}")
            lines.append(f"  action:      {step.action}")
            lines.append(f"  observation: {step.observation}")
            lines.append("")
        return "\n".join(lines).rstrip() + ("\n" if self.steps else "")
