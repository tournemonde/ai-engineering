"""Unit tests for Session 6 stress metrics (pass / fail / boundary)."""

from __future__ import annotations

from evals.stress.metrics import CostBudgetMetric, LatencyBudgetMetric, MemoryDriftMetric


def test_latency_budget_pass() -> None:
    metric = LatencyBudgetMetric(budget_ms=4000)
    result = metric.evaluate({"latency_ms": 2500})
    assert result.passed is True
    assert result.score == 1.0


def test_latency_budget_fail() -> None:
    metric = LatencyBudgetMetric(budget_ms=4000)
    result = metric.evaluate({"latency_ms": 4001})
    assert result.passed is False
    assert result.score == 0.0


def test_latency_budget_boundary_inclusive() -> None:
    metric = LatencyBudgetMetric(budget_ms=4000)
    result = metric.evaluate({"latency_ms": 4000})
    assert result.passed is True


def test_cost_budget_pass() -> None:
    metric = CostBudgetMetric(budget_usd=0.05)
    result = metric.evaluate({"cost_usd": 0.01})
    assert result.passed is True


def test_cost_budget_fail() -> None:
    metric = CostBudgetMetric(budget_usd=0.05)
    result = metric.evaluate({"cost_usd": 0.06})
    assert result.passed is False


def test_cost_budget_boundary_inclusive() -> None:
    metric = CostBudgetMetric(budget_usd=0.05)
    result = metric.evaluate({"cost_usd": 0.05})
    assert result.passed is True


def test_memory_drift_pass_in_metadata() -> None:
    metric = MemoryDriftMetric(fact="project name: Nimbus")
    result = metric.evaluate(
        {
            "summary": None,
            "anchors": [],
            "metadata": {"project_name": "Nimbus", "mentioned_technologies": ["React"]},
        }
    )
    assert result.passed is True
    assert result.score == 1.0


def test_memory_drift_pass_stack_includes_from_tech_list() -> None:
    metric = MemoryDriftMetric(fact="stack includes Flutter")
    result = metric.evaluate(
        {
            "summary": None,
            "anchors": [],
            "metadata": {"mentioned_technologies": ["React", "Flutter"]},
        }
    )
    assert result.passed is True


def test_memory_drift_fail() -> None:
    metric = MemoryDriftMetric(fact="budget locked: 30000 EUR")
    result = metric.evaluate(
        {
            "summary": "Earlier discussion of Helios dashboards.",
            "anchors": ["Okta SSO required"],
            "metadata": {"project_name": "Helios", "agreed_scope": "dashboards only"},
        }
    )
    assert result.passed is False


def test_memory_drift_boundary_case_insensitive_in_summary() -> None:
    metric = MemoryDriftMetric(fact="budget locked: 30000 EUR")
    result = metric.evaluate(
        {
            "summary": "Client said BUDGET LOCKED: 30000 EUR for v1.",
            "anchors": [],
            "metadata": {},
        }
    )
    assert result.passed is True
