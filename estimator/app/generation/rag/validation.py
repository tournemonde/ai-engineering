"""Post-generation checks for grounded estimates (Session 9 + Session 11).

Two independent guards run after the LLM returns an :class:`Estimate`:

* :func:`validate_citations` / :func:`verify_citations` — every cited chunk id
  must correspond to a chunk that was actually retrieved. Fabricated ids are
  the classic grounding failure and trigger one corrective retry in the
  orchestrator. Session 11 adds a structured :class:`CitationReport` that
  classifies each task line as grounded, dangling, or insufficient.
* :func:`check_coherence` — the ``insufficient`` confidence level has a strict
  shape (no numbers, an explanation present); a violation is a malformed
  response, not a valid estimate.
"""

from __future__ import annotations

from typing import Literal

import structlog
from pydantic import BaseModel, Field

from app.generation.rag.schemas import Estimate, RetrievedChunk

log = structlog.get_logger()

CitationStatus = Literal["grounded", "dangling", "insufficient"]


class CitationLineReport(BaseModel):
    """Per-task citation verification outcome."""

    module: str
    task: str
    status: CitationStatus
    cited_chunk_ids: list[int] = Field(default_factory=list)
    dangling_chunk_ids: list[int] = Field(default_factory=list)


class CitationReport(BaseModel):
    """Aggregate citation verification for one estimate.

    Distinguishes correctly grounded lines, dangling citations (invented ids)
    and lines explicitly marked as without sufficient source data.
    """

    grounded_lines: list[CitationLineReport] = Field(default_factory=list)
    dangling_lines: list[CitationLineReport] = Field(default_factory=list)
    insufficient_lines: list[CitationLineReport] = Field(default_factory=list)
    fabricated_chunk_ids: list[int] = Field(default_factory=list)

    @property
    def has_dangling(self) -> bool:
        return bool(self.dangling_lines or self.fabricated_chunk_ids)


def _task_cited_ids(estimate: Estimate) -> set[int]:
    """Collect every chunk id cited at task level."""
    cited: set[int] = set()
    for module in estimate.modules:
        for task in module.tasks:
            cited.update(ref.chunk_id for ref in task.sources)
    return cited


def _top_level_cited_ids(estimate: Estimate) -> set[int]:
    """Collect every chunk id cited at the estimate-level SourceCitation list."""
    return {citation.source_id for citation in estimate.sources}


def verify_citations(
    estimate: Estimate,
    retrieved_chunk_ids: set[int],
    *,
    request_id: str | None = None,
) -> CitationReport:
    """Flag any line whose cited chunk_id was never in the retrieved context.

    Parameters
    ----------
    estimate:
        The generated estimate to inspect.
    retrieved_chunk_ids:
        Ids of chunks that were actually passed to the LLM.
    request_id:
        Optional correlation id for structured logging.

    Returns
    -------
    CitationReport
        Per-line classification plus the sorted set of fabricated ids.
    """
    grounded_lines: list[CitationLineReport] = []
    dangling_lines: list[CitationLineReport] = []
    insufficient_lines: list[CitationLineReport] = []
    fabricated: set[int] = set()

    for module in estimate.modules:
        for task in module.tasks:
            cited = [ref.chunk_id for ref in task.sources]
            dangling = sorted({cid for cid in cited if cid not in retrieved_chunk_ids})
            fabricated.update(dangling)

            if not task.grounded:
                line = CitationLineReport(
                    module=module.name,
                    task=task.name,
                    status="insufficient",
                    cited_chunk_ids=cited,
                )
                insufficient_lines.append(line)
            elif dangling:
                line = CitationLineReport(
                    module=module.name,
                    task=task.name,
                    status="dangling",
                    cited_chunk_ids=cited,
                    dangling_chunk_ids=dangling,
                )
                dangling_lines.append(line)
            else:
                line = CitationLineReport(
                    module=module.name,
                    task=task.name,
                    status="grounded",
                    cited_chunk_ids=cited,
                )
                grounded_lines.append(line)

    # Top-level SourceCitation ids also count as fabricated when unknown.
    for sid in _top_level_cited_ids(estimate):
        if sid not in retrieved_chunk_ids:
            fabricated.add(sid)

    report = CitationReport(
        grounded_lines=grounded_lines,
        dangling_lines=dangling_lines,
        insufficient_lines=insufficient_lines,
        fabricated_chunk_ids=sorted(fabricated),
    )

    log.info(
        "citation_verification",
        request_id=request_id,
        grounded=len(grounded_lines),
        dangling=len(dangling_lines),
        insufficient=len(insufficient_lines),
        fabricated_chunk_ids=report.fabricated_chunk_ids,
        has_dangling=report.has_dangling,
    )
    return report


def validate_citations(
    estimate: Estimate,
    retrieved_chunks: list[RetrievedChunk],
    *,
    request_id: str | None = None,
) -> list[int]:
    """Return the cited source ids that were never retrieved (fabricated).

    Thin wrapper over :func:`verify_citations` kept for Session 9 call sites
    (orchestrator retry, stage generate). An empty list means every citation is
    valid (including the edge case of an estimate that cites nothing at all).
    """
    valid_ids = {chunk.id for chunk in retrieved_chunks}
    report = verify_citations(estimate, valid_ids, request_id=request_id)
    return report.fabricated_chunk_ids


def check_coherence(estimate: Estimate) -> bool:
    """Return whether the estimate's confidence level matches its content.

    When ``confidence == "insufficient"``: both numeric totals must be ``None``,
    ``modules`` must be empty, and ``insufficient_context_explanation`` must be
    non-empty. Any other confidence level is always considered coherent here
    (the numeric checks belong to the schema/business rules, not to this guard).
    """
    if estimate.confidence != "insufficient":
        return True
    return (
        estimate.total_engineer_days is None
        and estimate.duration_weeks is None
        and not estimate.modules
        and bool(estimate.insufficient_context_explanation)
    )
