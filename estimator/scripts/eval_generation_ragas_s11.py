#!/usr/bin/env python3
"""Session 11 generation-quality baseline with RAGAS.

For each of Q1–Q5 in ``evals/golden_generation_s11.json``:

1. Build a synthetic retrieved context from the relevant historical budgets
   (no Postgres required — corpus JSON only).
2. Run the real grounded generator (``generate_estimate`` via Instructor /
   ``LLMWrapper``).
3. Run ``verify_citations`` and keep the report for the first query.
4. Score with RAGAS: faithfulness, answer_relevancy, context_precision,
   context_recall.

Judge config: OpenAI chat model (``PRIMARY_MODEL`` / gpt-4o-mini fallback) +
``text-embedding-3-small``.

Usage (needs OPENAI_API_KEY)::

    cd estimator && uv run python scripts/eval_generation_ragas_s11.py

Writes:
* ``evals/session11_ragas_table.md`` — metrics table + short note
* ``evals/citation_report_from_generation.json`` — verify_citations on Q1
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Load .env from estimator/ and repo root without printing secrets.
for env_path in (ROOT / ".env", ROOT.parent / ".env"):
    if env_path.exists():
        from dotenv import load_dotenv

        load_dotenv(env_path)

# Prefer a fast structured model for the harness (gpt-5 often times out at 30s).
# Force (not setdefault): .env may pin GENERATION_MODEL=gpt-5 / LLM_TIMEOUT=30.
os.environ["GENERATION_MODEL"] = os.getenv("RAGAS_GENERATION_MODEL", "gpt-4o-mini")
os.environ["LLM_TIMEOUT"] = os.getenv("RAGAS_LLM_TIMEOUT", "120")
os.environ["GENERATION_REASONING_EFFORT"] = os.getenv("RAGAS_REASONING_EFFORT", "low")
os.environ.setdefault("RAGAS_JUDGE_MODEL", "gpt-4o-mini")

from datasets import Dataset  # noqa: E402
from openai import OpenAI  # noqa: E402
from ragas import evaluate  # noqa: E402
from ragas.embeddings import OpenAIEmbeddings  # noqa: E402
from ragas.llms import llm_factory  # noqa: E402
from ragas.metrics import (  # noqa: E402
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)

from app.config import get_settings  # noqa: E402
from app.dependencies import get_llm_wrapper  # noqa: E402
from app.generation.rag.context_assembler import build_context_block  # noqa: E402
from app.generation.rag.prompt_builder import (  # noqa: E402
    build_system_prompt,
    build_user_message,
)
from app.generation.rag.schemas import (  # noqa: E402
    Estimate,
    EstimationQuery,
    RetrievedChunk,
)
from app.generation.rag.validation import verify_citations  # noqa: E402

# Settings / wrapper are cached singletons — rebuild after env overrides.
get_settings.cache_clear()
if hasattr(get_llm_wrapper, "cache_clear"):
    get_llm_wrapper.cache_clear()


async def _generate_estimate(
    context_block: str, structured_query: EstimationQuery
) -> Estimate:
    """Generate via LLMWrapper without gpt-5 ``reasoning_effort`` (breaks gpt-4o-mini)."""
    settings = get_settings()
    wrapper = get_llm_wrapper()
    estimate, _meta = await asyncio.to_thread(
        wrapper.complete_structured,
        system_prompt=build_system_prompt(include_hours=True),
        user_message=build_user_message(context_block, structured_query),
        response_model=Estimate,
        model_override=settings.GENERATION_MODEL,
        reasoning_effort=None,
        max_tokens=min(settings.GENERATION_MAX_TOKENS, 8000),
    )
    return estimate

GOLDEN_PATH = ROOT / "evals" / "golden_generation_s11.json"
BUDGETS_PATH = ROOT / "data" / "budgets_sample.json"
TABLE_PATH = ROOT / "evals" / "session11_ragas_table.md"
CITATION_PATH = ROOT / "evals" / "citation_report_from_generation.json"


def _chunks_from_budgets(
    budget_ids: list[str], budgets: dict[str, dict]
) -> list[RetrievedChunk]:
    """Turn relevant budget components into citeable RetrievedChunk objects."""
    chunks: list[RetrievedChunk] = []
    next_id = 1
    for bid in budget_ids:
        budget = budgets[bid]
        sector = budget["client_metadata"]["sector"]
        year = int(budget["year"])
        for component in budget["components"]:
            content = (
                f"Budget: {bid}\n"
                f"Component: {component['name']}\n"
                f"{component['description']}\n"
                f"Estimated hours: {component['estimated_hours']}\n"
                f"Tech: {', '.join(component.get('tech_stack', []))}"
            )
            chunks.append(
                RetrievedChunk(
                    id=next_id,
                    content=content,
                    sector=sector,
                    project_year=year,
                    chunk_type="budget_component",
                    distance=0.2,
                    budget_id=bid,
                    source_id=bid,
                )
            )
            next_id += 1
    return chunks


def _query_from_text(text: str, sector: str | None) -> EstimationQuery:
    """Minimal structured brief when we skip reformulation to save tokens."""
    return EstimationQuery(
        function=text,
        technologies=[],
        sector=sector,
        scale="medium",
        country="ES",
        regulations=[],
        constraints=[],
    )


def _sector_for(budget_ids: list[str], budgets: dict[str, dict]) -> str | None:
    sectors = {budgets[b]["client_metadata"]["sector"] for b in budget_ids}
    return next(iter(sectors)) if len(sectors) == 1 else None


def _answer_text(estimate) -> str:
    """Flatten the structured estimate into judge-readable prose."""
    lines = [
        f"confidence={estimate.confidence}",
        f"total_engineer_days={estimate.total_engineer_days}",
        f"duration_weeks={estimate.duration_weeks}",
        f"reasoning={estimate.reasoning}",
    ]
    for module in estimate.modules:
        lines.append(f"module={module.name}")
        for task in module.tasks:
            refs = ", ".join(
                f"{ref.document_id}#{ref.chunk_id}" for ref in task.sources
            )
            lines.append(
                f"  task={task.name} days={task.engineer_days} "
                f"grounded={task.grounded} sources=[{refs}]"
            )
            for ref in task.sources:
                lines.append(f"    evidence={ref.evidence[:200]}")
    for assumption in estimate.assumptions:
        lines.append(f"assumption={assumption.description} ({assumption.impact})")
    if estimate.insufficient_context_explanation:
        lines.append(f"insufficient={estimate.insufficient_context_explanation}")
    return "\n".join(lines)


async def _generate_rows(golden: dict, budgets: dict[str, dict]) -> list[dict]:
    rows: list[dict] = []
    for item in golden["queries"]:
        budget_ids = item["relevant_budget_ids"]
        chunks = _chunks_from_budgets(budget_ids, budgets)
        context_block = build_context_block(chunks)
        structured = _query_from_text(item["query"], _sector_for(budget_ids, budgets))
        print(f"Generating {item['id']} over {len(chunks)} chunks…", flush=True)
        estimate = await _generate_estimate(context_block, structured)
        report = verify_citations(
            estimate,
            {c.id for c in chunks},
            request_id=f"ragas-{item['id']}",
        )
        if item["id"] == "Q1":
            CITATION_PATH.write_text(
                json.dumps(
                    {
                        "query_id": "Q1",
                        "retrieved_chunk_ids": [c.id for c in chunks],
                        "report": report.model_dump(),
                        "estimate_summary": {
                            "confidence": estimate.confidence,
                            "total_engineer_days": estimate.total_engineer_days,
                            "modules": len(estimate.modules),
                            "tasks": sum(len(m.tasks) for m in estimate.modules),
                        },
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            print(f"Wrote citation report → {CITATION_PATH}", flush=True)

        rows.append(
            {
                "id": item["id"],
                "user_input": item["query"],
                "response": _answer_text(estimate),
                "retrieved_contexts": [c.content for c in chunks],
                "reference": item["ground_truth"],
                "citation_dangling": report.fabricated_chunk_ids,
            }
        )
    return rows


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def _write_table(rows: list[dict], scores: dict[str, list[float]]) -> None:
    metrics = [
        "faithfulness",
        "answer_relevancy",
        "context_precision",
        "context_recall",
    ]
    lines = [
        "# Session 11 — RAGAS generation-quality baseline",
        "",
        "Judge: OpenAI chat + `text-embedding-3-small`. Contexts = relevant "
        "budget components from `data/budgets_sample.json`. Answers = "
        "`generate_estimate` (Instructor / LLMWrapper).",
        "",
        "| Query | faithfulness | answer_relevancy | context_precision | context_recall |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for i, row in enumerate(rows):
        cells = [f"{scores[m][i]:.3f}" for m in metrics]
        lines.append(f"| {row['id']} | " + " | ".join(cells) + " |")
    avgs = [f"{_mean(scores[m]):.3f}" for m in metrics]
    lines.append("| **average** | " + " | ".join(avgs) + " |")
    lines.extend(
        [
            "",
            "## Note (2–3 sentences)",
            "",
            "Los números más llamativos suelen ser faithfulness vs context_recall: "
            "si el generador cita bien los chunks pero resume en engineer-days "
            "agregados, faithfulness puede bajar porque el juez no ve la cifra "
            "exacta en el contexto (horas vs días). Context_recall depende de "
            "que el ground_truth mencione análogos concretos (BUD-*) presentes "
            "en los contextos; Q2 es sensible a distractores marketplace/grocery. "
            "Traer esta tabla al directo como baseline antes de medir detección "
            "de alucinaciones.",
            "",
        ]
    )
    TABLE_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nWrote {TABLE_PATH}")


async def main() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is required to run the RAGAS baseline.")

    golden = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
    budgets_list = json.loads(BUDGETS_PATH.read_text(encoding="utf-8"))
    budgets = {b["budget_id"]: b for b in budgets_list}

    rows = await _generate_rows(golden, budgets)

    dataset = Dataset.from_list(
        [
            {
                "user_input": r["user_input"],
                "response": r["response"],
                "retrieved_contexts": r["retrieved_contexts"],
                "reference": r["reference"],
            }
            for r in rows
        ]
    )

    settings = get_settings()
    judge_model = os.getenv("RAGAS_JUDGE_MODEL") or getattr(
        settings, "PRIMARY_MODEL", None
    ) or "gpt-4o-mini"
    # Strip provider prefixes like "openai/" if present for llm_factory.
    if "/" in judge_model:
        judge_model = judge_model.split("/", 1)[-1]

    print(f"Running RAGAS with judge={judge_model}…", flush=True)
    openai_client = OpenAI()
    llm = llm_factory(judge_model, client=openai_client)
    embeddings = OpenAIEmbeddings(client=openai_client, model="text-embedding-3-small")
    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=llm,
        embeddings=embeddings,
    )
    df = result.to_pandas()
    scores = {
        "faithfulness": [float(x) for x in df["faithfulness"].tolist()],
        "answer_relevancy": [float(x) for x in df["answer_relevancy"].tolist()],
        "context_precision": [float(x) for x in df["context_precision"].tolist()],
        "context_recall": [float(x) for x in df["context_recall"].tolist()],
    }
    _write_table(rows, scores)


if __name__ == "__main__":
    asyncio.run(main())
