#!/usr/bin/env python3
"""Run the Session 12 hand-written estimation agent and print its trace.

Examples::

    # Cheap loop debug with the offline stub (no DB)
    uv run python scripts/run_agent_s12.py \\
        exercises/session-12/sample_transcript_simple.txt \\
        --model gpt-5-mini --stub

    # Deliverable run on the complex transcript
    uv run python scripts/run_agent_s12.py \\
        exercises/session-12/sample_transcript_complex.txt \\
        --model gpt-5 --effort medium \\
        --out exercises/session-12/example_trace_complex.txt
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

# Allow `uv run python scripts/run_agent_s12.py` from estimator/
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


async def _main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("transcript", type=Path, help="Path to the transcript .txt")
    parser.add_argument("--model", default=None, help="Override AGENT_MODEL")
    parser.add_argument("--effort", default=None, help="Override AGENT_REASONING_EFFORT")
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=None,
        help="Override AGENT_MAX_ITERATIONS",
    )
    parser.add_argument(
        "--stub",
        action="store_true",
        help="Use exercises/session-12/reference_retrieval.py (no DB)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Write the rendered trace (+ estimate summary) to this file",
    )
    args = parser.parse_args()

    if not args.transcript.is_file():
        print(f"Transcript not found: {args.transcript}", file=sys.stderr)
        return 2

    from app.dependencies import get_async_openai_client
    from app.generation.agentic.agent_loop import run_estimation_agent

    client = get_async_openai_client()
    if client is None:
        print("OPENAI_API_KEY is not set — cannot run the agent.", file=sys.stderr)
        return 2

    retrieval_backend = None
    if args.stub:
        from importlib.util import module_from_spec, spec_from_file_location

        stub_path = _ROOT / "exercises" / "session-12" / "reference_retrieval.py"
        spec = spec_from_file_location("s12_reference_retrieval", stub_path)
        assert spec and spec.loader
        mod = module_from_spec(spec)
        spec.loader.exec_module(mod)

        async def retrieval_backend(query: str, filters: dict | None = None):
            return mod.search_budgets_stub(query, filters)

    transcript = args.transcript.read_text(encoding="utf-8")
    estimate, trace = await run_estimation_agent(
        transcript,
        client=client,
        model=args.model,
        effort=args.effort,
        max_iterations=args.max_iterations,
        retrieval_backend=retrieval_backend,
    )

    rendered = trace.render()
    summary = (
        "\n=== ESTIMATE ===\n"
        f"total_hours={estimate.total_hours}\n"
        f"confidence={estimate.confidence}\n"
        f"notes={estimate.notes}\n"
        "components:\n"
        + "\n".join(
            f"  - {c.name}: {c.hours}h"
            f" refs={c.reference_budget_ids or '-'} unbudgeted={c.unbudgeted}"
            for c in estimate.components
        )
        + "\n"
    )
    output = rendered + "\n" + summary
    print(output)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(output, encoding="utf-8")
        print(f"Wrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_main()))
