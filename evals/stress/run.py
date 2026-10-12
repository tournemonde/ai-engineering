"""CLI stress runner for the conversational CAG.

Usage::

    uv run python -m evals.stress.run --http http://localhost:8000 \\
        --scenarios growing,pivot,contradiction \\
        --attachment-sizes 0,5,20,50,100 \\
        --repeats 3 \\
        --output evals/stress/results.csv

Default transport is in-process ``TestClient``. ``--http`` targets a live server.
``turn_observed`` is read from ``GET /sessions/{id}.last_turn`` (no log scraping).
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Protocol

import fakeredis
import httpx
from fastapi.testclient import TestClient

from app.config import get_settings
from app.dependencies import (
    get_estimation_service,
    get_llm_wrapper,
    get_openai_client,
    get_semantic_cache,
    get_session_store,
)
from app.main import app
from app.services.cache import EstimationCache
from app.services.estimation import EstimationService
from app.services.llm_wrapper import LLMWrapper
from app.sessions.store import SessionStore
from evals.stress.fixtures.build_pdfs import MARKERS, build_all
from evals.stress.metrics import CostBudgetMetric, LatencyBudgetMetric, MemoryDriftMetric
from evals.stress.scenarios import SCENARIOS, ScenarioProfile, get_scenario

DEFAULT_FORM = {
    "project_type": "web_saas",
    "detail_level": "medium",
    "output_format": "phases_table",
}

CSV_COLUMNS = [
    "scenario",
    "repeat",
    "attachment_size_kb",
    "turn_index",
    "session_id",
    "enriched_transcript_chars",
    "attachments_total_chars",
    "messages_in_window",
    "anchors_count",
    "summary_chars",
    "tokens_in",
    "tokens_out",
    "cost_usd",
    "latency_ms",
    "cache_hit_kind",
    "last_resolved_tier",
    "latency_budget_pass",
    "cost_budget_pass",
    "memory_drift_pass",
    "memory_drift_fact",
    "error",
]


class StressClient(Protocol):
    def create_session(self) -> str: ...

    def estimate(
        self,
        session_id: str,
        *,
        transcript: str,
        attachment_path: Path | None,
    ) -> dict[str, Any]: ...

    def get_session(self, session_id: str) -> dict[str, Any]: ...

    def close(self) -> None: ...


def _wire_inprocess_live(*, primary_model: str | None = None) -> None:
    """Wire real LLM keys with fakeredis so stress runs without Docker Redis.

    Conversational estimates skip caches, but ``LLMWrapper`` still requires an
    ``EstimationCache`` instance. Fakeredis keeps that dependency local.
    """
    if primary_model:
        os.environ["PRIMARY_MODEL"] = primary_model
    # Force localhost Redis URL out of the way even if Settings reads it later.
    os.environ.setdefault("REDIS_URL", "redis://127.0.0.1:6379")
    get_settings.cache_clear()
    settings = get_settings()

    cache = EstimationCache(
        fakeredis.FakeRedis(decode_responses=True),
        ttl=settings.CACHE_TTL,
    )
    wrapper = LLMWrapper(
        openai_api_key=settings.OPENAI_API_KEY,
        anthropic_api_key=settings.ANTHROPIC_API_KEY,
        primary_model=settings.PRIMARY_MODEL,
        fallback_model=settings.FALLBACK_MODEL,
        timeout=settings.LLM_TIMEOUT,
        num_retries=settings.LLM_RETRIES,
        cache=cache,
    )
    store = SessionStore(max_turns=settings.MAX_CONVERSATION_TURNS)
    service = EstimationService(
        llm_wrapper=wrapper,
        exact_cache=cache,
        semantic_cache=None,
        openai_client=get_openai_client(),
        metadata_extractor_model=settings.METADATA_EXTRACTOR_MODEL,
        compression_model=settings.COMPRESSION_MODEL,
        anchor_detection_mode=settings.ANCHOR_DETECTION_MODE,
        conversational_prompt_version=settings.CONVERSATIONAL_PROMPT_VERSION,
        critic_model=settings.CRITIC_MODEL,
        boss_max_iterations=settings.BOSS_MAX_ITERATIONS,
    )
    app.dependency_overrides[get_estimation_service] = lambda: service
    app.dependency_overrides[get_session_store] = lambda: store
    app.dependency_overrides[get_llm_wrapper] = lambda: wrapper
    app.dependency_overrides[get_semantic_cache] = lambda: None


class InProcessClient:
    def __init__(self, *, primary_model: str | None = None) -> None:
        _wire_inprocess_live(primary_model=primary_model)
        self._client = TestClient(app)

    def create_session(self) -> str:
        response = self._client.post("/sessions")
        response.raise_for_status()
        return str(response.json()["session_id"])

    def estimate(
        self,
        session_id: str,
        *,
        transcript: str,
        attachment_path: Path | None,
    ) -> dict[str, Any]:
        data = {**DEFAULT_FORM, "transcript": transcript}
        files = None
        file_handle = None
        try:
            if attachment_path is not None:
                file_handle = attachment_path.open("rb")
                files = {
                    "attachments": (
                        attachment_path.name,
                        file_handle,
                        "application/pdf",
                    )
                }
            response = self._client.post(
                f"/sessions/{session_id}/estimate",
                data=data,
                files=files,
            )
        finally:
            if file_handle is not None:
                file_handle.close()
        response.raise_for_status()
        return dict(response.json())

    def get_session(self, session_id: str) -> dict[str, Any]:
        response = self._client.get(f"/sessions/{session_id}")
        response.raise_for_status()
        return dict(response.json())

    def close(self) -> None:
        self._client.close()
        app.dependency_overrides.clear()


class HttpClient:
    def __init__(self, base_url: str, timeout_s: float = 180.0) -> None:
        self._client = httpx.Client(base_url=base_url.rstrip("/"), timeout=timeout_s)

    def create_session(self) -> str:
        response = self._client.post("/sessions")
        response.raise_for_status()
        return str(response.json()["session_id"])

    def estimate(
        self,
        session_id: str,
        *,
        transcript: str,
        attachment_path: Path | None,
    ) -> dict[str, Any]:
        data = {**DEFAULT_FORM, "transcript": transcript}
        files = None
        file_handle = None
        try:
            if attachment_path is not None:
                file_handle = attachment_path.open("rb")
                files = {
                    "attachments": (
                        attachment_path.name,
                        file_handle,
                        "application/pdf",
                    )
                }
            response = self._client.post(
                f"/sessions/{session_id}/estimate",
                data=data,
                files=files,
            )
        finally:
            if file_handle is not None:
                file_handle.close()
        response.raise_for_status()
        return dict(response.json())

    def get_session(self, session_id: str) -> dict[str, Any]:
        response = self._client.get(f"/sessions/{session_id}")
        response.raise_for_status()
        return dict(response.json())

    def close(self) -> None:
        self._client.close()


def _pdf_path_for_size(size_kb: int, generated_dir: Path) -> Path | None:
    if size_kb <= 0:
        return None
    filename = f"attach_{size_kb}kb.pdf"
    path = generated_dir / filename
    if not path.exists():
        raise FileNotFoundError(f"Missing PDF fixture {path}; run build_pdfs first")
    return path


def _attachment_marker(size_kb: int) -> str | None:
    if size_kb <= 0:
        return None
    return MARKERS[f"attach_{size_kb}kb.pdf"]


def _evaluate_row(
    *,
    last_turn: dict[str, Any],
    snapshot: dict[str, Any],
    fact: str | None,
    latency_budget_ms: int,
    cost_budget_usd: float,
) -> dict[str, Any]:
    latency = LatencyBudgetMetric(budget_ms=latency_budget_ms).evaluate(last_turn)
    cost = CostBudgetMetric(budget_usd=cost_budget_usd).evaluate(last_turn)
    drift_pass = ""
    drift_fact = fact or ""
    if fact:
        drift = MemoryDriftMetric(fact=fact).evaluate(snapshot)
        drift_pass = "1" if drift.passed else "0"
    return {
        "latency_budget_pass": "1" if latency.passed else "0",
        "cost_budget_pass": "1" if cost.passed else "0",
        "memory_drift_pass": drift_pass,
        "memory_drift_fact": drift_fact,
    }


def _run_scenario_repeat(
    client: StressClient,
    *,
    scenario: ScenarioProfile,
    repeat: int,
    attachment_size_kb: int,
    attachment_path: Path | None,
    latency_budget_ms: int,
    cost_budget_usd: float,
    max_turns: int | None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    session_id = client.create_session()
    turns = scenario.turns
    if max_turns is not None:
        turns = turns[:max_turns]

    # Facts declared on earlier turns are checked on later snapshots.
    declared_facts: list[tuple[int, str]] = []

    for turn in turns:
        declared_facts.append((turn.turn_index, turn.fact_to_remember))
        error = ""
        last_exc: Exception | None = None
        for attempt in range(1, 4):
            try:
                client.estimate(
                    session_id,
                    transcript=turn.transcript,
                    attachment_path=attachment_path if turn.turn_index == 1 else None,
                )
                last_exc = None
                break
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                time.sleep(min(2 * attempt, 6))
        if last_exc is not None:
            error = f"{type(last_exc).__name__}: {last_exc}"[:400]
            rows.append(
                {
                    "scenario": scenario.name,
                    "repeat": repeat,
                    "attachment_size_kb": attachment_size_kb,
                    "turn_index": turn.turn_index,
                    "session_id": session_id,
                    "enriched_transcript_chars": "",
                    "attachments_total_chars": "",
                    "messages_in_window": "",
                    "anchors_count": "",
                    "summary_chars": "",
                    "tokens_in": "",
                    "tokens_out": "",
                    "cost_usd": "",
                    "latency_ms": "",
                    "cache_hit_kind": "",
                    "last_resolved_tier": "",
                    "latency_budget_pass": "",
                    "cost_budget_pass": "",
                    "memory_drift_pass": "",
                    "memory_drift_fact": declared_facts[0][1] if declared_facts else "",
                    "error": error,
                }
            )
            # Stop this conversation on hard failure; continue other repeats.
            break

        snapshot = client.get_session(session_id)
        last_turn = snapshot.get("last_turn") or {}
        # Prefer earliest durable fact (project name / first budget) for drift.
        fact = declared_facts[0][1] if declared_facts else turn.fact_to_remember
        metrics = _evaluate_row(
            last_turn=last_turn,
            snapshot=snapshot,
            fact=fact,
            latency_budget_ms=latency_budget_ms,
            cost_budget_usd=cost_budget_usd,
        )
        rows.append(
            {
                "scenario": scenario.name,
                "repeat": repeat,
                "attachment_size_kb": attachment_size_kb,
                "turn_index": last_turn.get("turn_index", turn.turn_index),
                "session_id": session_id,
                "enriched_transcript_chars": last_turn.get("enriched_transcript_chars", ""),
                "attachments_total_chars": last_turn.get("attachments_total_chars", ""),
                "messages_in_window": last_turn.get("messages_in_window", ""),
                "anchors_count": last_turn.get("anchors_count", ""),
                "summary_chars": last_turn.get("summary_chars", ""),
                "tokens_in": last_turn.get("tokens_in", ""),
                "tokens_out": last_turn.get("tokens_out", ""),
                "cost_usd": last_turn.get("cost_usd", ""),
                "latency_ms": last_turn.get("latency_ms", ""),
                "cache_hit_kind": last_turn.get("cache_hit_kind", ""),
                "last_resolved_tier": last_turn.get("last_resolved_tier", ""),
                **metrics,
                "error": error,
            }
        )
    return rows


def _run_attachment_probe(
    client: StressClient,
    *,
    size_kb: int,
    repeat: int,
    attachment_path: Path | None,
    latency_budget_ms: int,
    cost_budget_usd: float,
) -> list[dict[str, Any]]:
    """Single-turn attachment size probe (same short transcript)."""
    transcript = (
        "Estimate a short internal reporting dashboard called Helios Lite "
        "for finance with Next.js and Postgres. Keep the estimate concise."
    )
    marker = _attachment_marker(size_kb)
    session_id = client.create_session()
    try:
        client.estimate(
            session_id,
            transcript=transcript,
            attachment_path=attachment_path,
        )
        snapshot = client.get_session(session_id)
        last_turn = snapshot.get("last_turn") or {}
        metrics = _evaluate_row(
            last_turn=last_turn,
            snapshot=snapshot,
            fact=marker,
            latency_budget_ms=latency_budget_ms,
            cost_budget_usd=cost_budget_usd,
        )
        return [
            {
                "scenario": "attachment_probe",
                "repeat": repeat,
                "attachment_size_kb": size_kb,
                "turn_index": last_turn.get("turn_index", 1),
                "session_id": session_id,
                "enriched_transcript_chars": last_turn.get("enriched_transcript_chars", ""),
                "attachments_total_chars": last_turn.get("attachments_total_chars", ""),
                "messages_in_window": last_turn.get("messages_in_window", ""),
                "anchors_count": last_turn.get("anchors_count", ""),
                "summary_chars": last_turn.get("summary_chars", ""),
                "tokens_in": last_turn.get("tokens_in", ""),
                "tokens_out": last_turn.get("tokens_out", ""),
                "cost_usd": last_turn.get("cost_usd", ""),
                "latency_ms": last_turn.get("latency_ms", ""),
                "cache_hit_kind": last_turn.get("cache_hit_kind", ""),
                "last_resolved_tier": last_turn.get("last_resolved_tier", ""),
                **metrics,
                "error": "",
            }
        ]
    except Exception as exc:  # noqa: BLE001
        return [
            {
                "scenario": "attachment_probe",
                "repeat": repeat,
                "attachment_size_kb": size_kb,
                "turn_index": 1,
                "session_id": session_id,
                "enriched_transcript_chars": "",
                "attachments_total_chars": "",
                "messages_in_window": "",
                "anchors_count": "",
                "summary_chars": "",
                "tokens_in": "",
                "tokens_out": "",
                "cost_usd": "",
                "latency_ms": "",
                "cache_hit_kind": "",
                "last_resolved_tier": "",
                "latency_budget_pass": "",
                "cost_budget_pass": "",
                "memory_drift_pass": "",
                "memory_drift_fact": marker or "",
                "error": f"{type(exc).__name__}: {exc}"[:400],
            }
        ]


def run_stress(
    *,
    client: StressClient,
    scenarios: list[ScenarioProfile],
    attachment_sizes: list[int],
    repeats: int,
    generated_dir: Path,
    latency_budget_ms: int,
    cost_budget_usd: float,
    max_turns: int | None,
    include_multi_turn: bool,
    include_attachments: bool,
    workers: int = 3,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    jobs: list[tuple[str, Any]] = []

    if include_multi_turn:
        for scenario in scenarios:
            for repeat in range(1, repeats + 1):
                jobs.append(
                    (
                        f"multi-turn:{scenario.name}:r{repeat}",
                        lambda sc=scenario, rp=repeat: _run_scenario_repeat(
                            client,
                            scenario=sc,
                            repeat=rp,
                            attachment_size_kb=0,
                            attachment_path=None,
                            latency_budget_ms=latency_budget_ms,
                            cost_budget_usd=cost_budget_usd,
                            max_turns=max_turns,
                        ),
                    )
                )

    if include_attachments:
        needed = [s for s in attachment_sizes if s > 0]
        if needed:
            build_all(output_dir=generated_dir)
        for size_kb in attachment_sizes:
            path = _pdf_path_for_size(size_kb, generated_dir)
            for repeat in range(1, repeats + 1):
                jobs.append(
                    (
                        f"attachment:{size_kb}kb:r{repeat}",
                        lambda sk=size_kb, rp=repeat, p=path: _run_attachment_probe(
                            client,
                            size_kb=sk,
                            repeat=rp,
                            attachment_path=p,
                            latency_budget_ms=latency_budget_ms,
                            cost_budget_usd=cost_budget_usd,
                        ),
                    )
                )

    # Independent sessions can run in parallel; turns within a session stay serial.
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = {pool.submit(fn): label for label, fn in jobs}
        for fut in as_completed(futures):
            label = futures[fut]
            print(f"[done] {label}")
            rows.extend(fut.result())

    rows.sort(
        key=lambda r: (
            str(r.get("scenario", "")),
            int(r.get("attachment_size_kb") or 0),
            int(r.get("repeat") or 0),
            int(r.get("turn_index") or 0),
        )
    )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in CSV_COLUMNS})


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CAG stress runner (Session 6)")
    parser.add_argument(
        "--http",
        default=None,
        help="Base URL of a live FastAPI server. Omit for in-process TestClient.",
    )
    parser.add_argument(
        "--scenarios",
        default="growing,pivot,contradiction",
        help="Comma-separated scenario names",
    )
    parser.add_argument(
        "--attachment-sizes",
        default="0,5,20,50,100",
        help="Comma-separated attachment sizes in KB (0 = no attachment)",
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path, default=Path("evals/stress/results.csv"))
    parser.add_argument("--latency-budget-ms", type=int, default=4000)
    parser.add_argument("--cost-budget-usd", type=float, default=0.05)
    parser.add_argument(
        "--max-turns",
        type=int,
        default=None,
        help="Cap multi-turn conversations (default: all 20 turns)",
    )
    parser.add_argument(
        "--skip-multi-turn",
        action="store_true",
        help="Only run attachment probes",
    )
    parser.add_argument(
        "--skip-attachments",
        action="store_true",
        help="Only run multi-turn scenarios",
    )
    parser.add_argument(
        "--fixtures-dir",
        type=Path,
        default=Path("evals/stress/fixtures/generated"),
    )
    parser.add_argument(
        "--primary-model",
        default="gpt-4o-mini",
        help="Override PRIMARY_MODEL for in-process runs (keeps curves comparable).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Parallel sessions (default 1: SessionStore is not thread-safe).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    scenario_names = [s.strip() for s in args.scenarios.split(",") if s.strip()]
    scenarios = [get_scenario(name) for name in scenario_names]
    attachment_sizes = [int(x) for x in args.attachment_sizes.split(",") if x.strip()]

    unknown = [n for n in scenario_names if n not in SCENARIOS]
    if unknown:
        print(f"Unknown scenarios: {unknown}", file=sys.stderr)
        return 2

    client: StressClient
    if args.http:
        client = HttpClient(args.http)
    else:
        client = InProcessClient(primary_model=args.primary_model)

    t0 = time.perf_counter()
    try:
        rows = run_stress(
            client=client,
            scenarios=scenarios,
            attachment_sizes=attachment_sizes,
            repeats=args.repeats,
            generated_dir=args.fixtures_dir,
            latency_budget_ms=args.latency_budget_ms,
            cost_budget_usd=args.cost_budget_usd,
            max_turns=args.max_turns,
            include_multi_turn=not args.skip_multi_turn,
            include_attachments=not args.skip_attachments,
            workers=args.workers,
        )
    finally:
        client.close()

    write_csv(args.output, rows)
    elapsed = time.perf_counter() - t0
    errors = sum(1 for r in rows if r.get("error"))
    print(f"Wrote {len(rows)} rows to {args.output} ({errors} errors) in {elapsed:.1f}s")
    return 0 if errors == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
