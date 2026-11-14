#!/usr/bin/env python3
"""Produce a Session 11 citation-verification report (deterministic demo).

Builds one estimate with a correctly grounded task, an intentional dangling
citation, and an insufficient (ungrounded) task — then runs
``verify_citations`` and writes the JSON report.

Usage::

    uv run python scripts/demo_citation_verification_s11.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.generation.rag.schemas import (  # noqa: E402
    Assumption,
    Estimate,
    SourceCitation,
    SourceReference,
    TaskItem,
    WorkModule,
)
from app.generation.rag.validation import verify_citations  # noqa: E402

OUT = ROOT / "evals" / "citation_report_sample.json"


def main() -> None:
    retrieved_chunk_ids = {101, 102}
    estimate = Estimate(
        total_engineer_days=40,
        duration_weeks=8,
        modules=[
            WorkModule(
                name="Authentication",
                tasks=[
                    TaskItem(
                        name="OAuth2 backend",
                        engineer_days=15,
                        grounded=True,
                        sources=[
                            SourceReference(
                                chunk_id=101,
                                document_id="BUD-2024-001",
                                evidence="OAuth 2.0 authentication backend: 120h",
                            )
                        ],
                    ),
                    TaskItem(
                        name="Invented biometric login",
                        engineer_days=10,
                        grounded=True,
                        sources=[
                            SourceReference(
                                chunk_id=999,
                                document_id="BUD-FAKE",
                                evidence="biometrics (fabricated)",
                            )
                        ],
                    ),
                    TaskItem(
                        name="Novel fraud ML model",
                        engineer_days=None,
                        grounded=False,
                        sources=[],
                    ),
                ],
            )
        ],
        sources=[
            SourceCitation(source_id=101, relevance="primary", used_for="oauth"),
            SourceCitation(source_id=999, relevance="supporting", used_for="ghost"),
        ],
        assumptions=[
            Assumption(
                description="Novel fraud ML model has no historical analog",
                impact="medium",
                rationale="Corpus has rules-based fraud, not custom ML.",
            )
        ],
        confidence="medium",
        reasoning="Demo estimate mixing grounded, dangling and insufficient lines.",
    )

    report = verify_citations(estimate, retrieved_chunk_ids, request_id="demo-s11")
    payload = {
        "description": (
            "Session 11 citation verification sample. Intentionally includes a "
            "dangling chunk_id=999 to prove detectability."
        ),
        "retrieved_chunk_ids": sorted(retrieved_chunk_ids),
        "report": report.model_dump(),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
