"""Unit tests for post-generation validation (Session 9 + Session 11)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.generation.rag.schemas import (
    Estimate,
    RetrievedChunk,
    SourceCitation,
    SourceReference,
    TaskItem,
    WorkModule,
)
from app.generation.rag.validation import (
    check_coherence,
    validate_citations,
    verify_citations,
)


def _chunk(chunk_id: int) -> RetrievedChunk:
    return RetrievedChunk(
        id=chunk_id,
        content="Component: Auth\nEstimated hours: 120",
        sector="finance",
        project_year=2024,
        chunk_type="budget_component",
        distance=0.3,
        budget_id="BUD-2024-001",
        source_id="BUD-2024-001",
    )


def _ref(chunk_id: int, *, document_id: str = "BUD-2024-001") -> SourceReference:
    return SourceReference(
        chunk_id=chunk_id,
        document_id=document_id,
        evidence="Estimated hours: 120",
    )


def _estimate(
    *,
    source_ids: list[int],
    component_sources: list[int],
    confidence="high",
    grounded: bool = True,
) -> Estimate:
    if grounded:
        task = TaskItem(
            name="Auth",
            engineer_days=20,
            grounded=True,
            sources=[_ref(cid) for cid in component_sources],
        )
    else:
        task = TaskItem(name="Auth", engineer_days=None, grounded=False, sources=[])
    return Estimate(
        total_engineer_days=20 if grounded else None,
        duration_weeks=4 if grounded else None,
        modules=[WorkModule(name="Authentication", tasks=[task])],
        sources=[
            SourceCitation(source_id=sid, relevance="primary", used_for="auth")
            for sid in source_ids
        ],
        assumptions=[],
        confidence=confidence,
        reasoning="Derived from retrieved budgets.",
    )


def test_validate_citations_all_valid_returns_empty():
    chunks = [_chunk(1), _chunk(2)]
    estimate = _estimate(source_ids=[1, 2], component_sources=[1])
    assert validate_citations(estimate, chunks) == []


def test_validate_citations_flags_fabricated_ids():
    chunks = [_chunk(1), _chunk(2)]
    # cites 99 (top-level) and 42 (inside a component) — neither retrieved.
    estimate = _estimate(source_ids=[1, 99], component_sources=[42])
    assert validate_citations(estimate, chunks) == [42, 99]


def test_validate_citations_no_sources_is_valid():
    chunks = [_chunk(1)]
    estimate = _estimate(source_ids=[], component_sources=[], grounded=False)
    assert validate_citations(estimate, chunks) == []


def test_validate_citations_empty_retrieval_flags_every_cited_id():
    estimate = _estimate(source_ids=[1], component_sources=[2])
    assert validate_citations(estimate, []) == [1, 2]


def test_verify_citations_classifies_grounded_dangling_and_insufficient():
    chunks = {_chunk(1).id, _chunk(2).id}
    estimate = Estimate(
        total_engineer_days=25,
        duration_weeks=5,
        modules=[
            WorkModule(
                name="Auth",
                tasks=[
                    TaskItem(
                        name="OAuth",
                        engineer_days=15,
                        grounded=True,
                        sources=[_ref(1)],
                    ),
                    TaskItem(
                        name="Ghost feature",
                        engineer_days=10,
                        grounded=True,
                        sources=[_ref(999, document_id="BUD-FAKE")],
                    ),
                    TaskItem(
                        name="Novel ML scorer",
                        engineer_days=None,
                        grounded=False,
                        sources=[],
                    ),
                ],
            )
        ],
        sources=[SourceCitation(source_id=1, relevance="primary", used_for="oauth")],
        assumptions=[],
        confidence="medium",
        reasoning="Mixed grounding.",
    )
    report = verify_citations(estimate, chunks, request_id="req-test")
    assert [line.task for line in report.grounded_lines] == ["OAuth"]
    assert [line.task for line in report.dangling_lines] == ["Ghost feature"]
    assert report.dangling_lines[0].dangling_chunk_ids == [999]
    assert [line.task for line in report.insufficient_lines] == ["Novel ML scorer"]
    assert report.fabricated_chunk_ids == [999]
    assert report.has_dangling is True


def test_taskitem_grounded_true_requires_sources():
    with pytest.raises(ValidationError, match="grounded=True"):
        TaskItem(name="Auth", engineer_days=10, grounded=True, sources=[])


def test_taskitem_grounded_false_forbids_invented_hours():
    with pytest.raises(ValidationError, match="engineer_days"):
        TaskItem(name="Auth", engineer_days=10, grounded=False, sources=[])


def test_taskitem_grounded_false_forbids_sources():
    with pytest.raises(ValidationError, match="grounded=False"):
        TaskItem(name="Auth", engineer_days=None, grounded=False, sources=[_ref(1)])


def test_check_coherence_insufficient_with_nulls_is_coherent():
    estimate = Estimate(
        total_engineer_days=None,
        duration_weeks=None,
        confidence="insufficient",
        reasoning="no sources",
        insufficient_context_explanation="No relevant budgets retrieved.",
    )
    assert check_coherence(estimate) is True


def test_check_coherence_insufficient_with_numbers_is_incoherent():
    estimate = Estimate(
        total_engineer_days=10,
        duration_weeks=2,
        confidence="insufficient",
        reasoning="contradiction",
        insufficient_context_explanation="",
    )
    assert check_coherence(estimate) is False


def test_check_coherence_non_insufficient_always_true():
    estimate = _estimate(source_ids=[1], component_sources=[1], confidence="low")
    assert check_coherence(estimate) is True
