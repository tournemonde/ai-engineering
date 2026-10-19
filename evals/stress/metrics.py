"""Stress-specific metrics over ``turn_observed`` / session snapshots.

Kept in ``evals/stress/metrics.py`` (not ``evals/metrics.py``) because they
depend on the Session 6 observation shape (``turn_observed`` + session
snapshot) rather than ``(GoldenCase, EstimationResult)``.
"""

from __future__ import annotations

from typing import Any

from evals.metrics import MetricResult


class LatencyBudgetMetric:
    """1.0 if ``latency_ms`` ≤ ``budget_ms``; 0.0 otherwise."""

    name = "latency_budget"

    def __init__(self, budget_ms: int) -> None:
        self.budget_ms = budget_ms

    def evaluate(self, observation: dict[str, Any]) -> MetricResult:
        latency_ms = int(observation.get("latency_ms") or 0)
        passed = latency_ms <= self.budget_ms
        return MetricResult(
            name=self.name,
            score=1.0 if passed else 0.0,
            passed=passed,
            details=(
                f"latency_ms={latency_ms} budget_ms={self.budget_ms}"
                if passed
                else f"latency_ms={latency_ms} exceeded budget_ms={self.budget_ms}"
            ),
        )


class CostBudgetMetric:
    """1.0 if ``cost_usd`` ≤ ``budget_usd``; 0.0 otherwise."""

    name = "cost_budget"

    def __init__(self, budget_usd: float) -> None:
        self.budget_usd = budget_usd

    def evaluate(self, observation: dict[str, Any]) -> MetricResult:
        cost_usd = float(observation.get("cost_usd") or 0.0)
        passed = cost_usd <= self.budget_usd
        return MetricResult(
            name=self.name,
            score=1.0 if passed else 0.0,
            passed=passed,
            details=(
                f"cost_usd={cost_usd} budget_usd={self.budget_usd}"
                if passed
                else f"cost_usd={cost_usd} exceeded budget_usd={self.budget_usd}"
            ),
        )


class MemoryDriftMetric:
    """1.0 if ``fact`` appears in summary, anchors, or metadata (case-insensitive)."""

    name = "memory_drift"

    def __init__(
        self,
        fact: str,
        where: list[str] | None = None,
    ) -> None:
        self.fact = fact
        self.where = where or ["summary", "anchors", "metadata"]

    def evaluate(self, session_snapshot: dict[str, Any]) -> MetricResult:
        haystacks: list[str] = []
        if "summary" in self.where:
            summary = session_snapshot.get("summary") or ""
            haystacks.append(str(summary))
        if "anchors" in self.where:
            anchors = session_snapshot.get("anchors") or []
            if isinstance(anchors, list):
                haystacks.extend(str(a) for a in anchors)
            else:
                haystacks.append(str(anchors))
        if "metadata" in self.where:
            metadata = session_snapshot.get("metadata") or {}
            if isinstance(metadata, dict):
                parts: list[str] = []
                for key, value in metadata.items():
                    if value is None:
                        continue
                    label = str(key).replace("_", " ")
                    if isinstance(value, list):
                        joined = ", ".join(str(v) for v in value)
                        parts.extend(str(v) for v in value)
                        parts.append(f"{label}: {joined}")
                        for item in value:
                            # Common fact phrasing: "stack includes Flutter"
                            parts.append(f"stack includes {item}")
                    else:
                        parts.append(str(value))
                        parts.append(f"{label}: {value}")
                haystacks.append(" ".join(parts))
            else:
                haystacks.append(str(metadata))

        blob = " ".join(haystacks).lower()
        needle = self.fact.lower()
        found = needle in blob
        return MetricResult(
            name=self.name,
            score=1.0 if found else 0.0,
            passed=found,
            details=(
                f"fact found in {self.where}"
                if found
                else f"fact {self.fact!r} missing from {self.where}"
            ),
        )
