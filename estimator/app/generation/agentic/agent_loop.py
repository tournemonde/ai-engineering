"""Hand-written reason→act→observe loop over the OpenAI Responses API (S12).

DELIBERATE EXCEPTION to the repo convention: every other LLM call goes through
``LLMWrapper``. This module drives ``client.responses.parse`` by hand so each
step is visible in the trace. Do not "fix" it to use LLMWrapper.
"""

from __future__ import annotations

import json
from typing import Any

import structlog
from openai import AsyncOpenAI

from app.config import get_settings
from app.generation.agentic.agent_schemas import AgentEstimate, AgentStep, AgentTrace
from app.generation.agentic.agent_tools import (
    TOOL_SCHEMAS,
    RetrievalBackend,
    dispatch_tool,
)

log = structlog.get_logger()

SYSTEM_PROMPT = """\
You are a software estimation agent. Given a meeting transcript, you:

1. Identify the DISTINCT software components to estimate (do not merge unrelated ones).
2. Call search_budgets ONCE PER component with a focused query.
3. Call calculate_estimate with every component and the reference_amounts you collected
   (use estimated_hours from the search hits; use [] if a component had no hits).
4. Call validate_estimate on the candidate, then produce the final structured answer.

Rules:
- Never invent historical amounts — only use numbers returned by search_budgets.
- Keep search queries SHORT (3–12 keywords). Prefer domain terms over stack details.
- Pass filters.sectors as null unless the transcript clearly states a sector; wrong
  sector filters return empty results.
- If a search returns weak/empty results, reformulate once; if still empty, mark that
  component unbudgeted (reference_amounts=[]) rather than guessing.
- Always run validate_estimate before the final answer.
- Prefer English for tool arguments; the transcript may be in Spanish.
"""


def _reasoning_summary(response: Any) -> str:
    """Pull reasoning summaries from a Responses API turn, if present."""
    parts: list[str] = []
    for item in getattr(response, "output", []) or []:
        if getattr(item, "type", None) != "reasoning":
            continue
        for block in getattr(item, "summary", None) or []:
            text = getattr(block, "text", None)
            if text:
                parts.append(text)
    if parts:
        return " ".join(parts)
    return "(no reasoning summary emitted)"


def _format_action(name: str, arguments: str) -> str:
    try:
        args = json.loads(arguments)
        compact = json.dumps(args, ensure_ascii=False, separators=(",", ":"))
    except json.JSONDecodeError:
        compact = arguments
    if len(compact) > 240:
        compact = compact[:237] + "..."
    return f"{name}({compact})"


def _observation_summary(raw_json: str) -> str:
    try:
        data = json.loads(raw_json)
    except json.JSONDecodeError:
        return raw_json[:300]
    if isinstance(data, dict):
        if "summary" in data:
            return str(data["summary"])
        if "error" in data:
            return f"error: {data['error']}"
        if "count" in data:
            return f"{data['count']} hit(s)"
    return raw_json[:300]


async def run_estimation_agent(
    transcript: str,
    *,
    client: AsyncOpenAI,
    model: str | None = None,
    effort: str | None = None,
    max_iterations: int | None = None,
    retrieval_backend: RetrievalBackend | None = None,
) -> tuple[AgentEstimate, AgentTrace]:
    """Run the manual agent loop; return (estimate, trace)."""
    settings = get_settings()
    model = model or settings.AGENT_MODEL
    effort = effort or settings.AGENT_REASONING_EFFORT
    max_iterations = max_iterations or settings.AGENT_MAX_ITERATIONS

    trace = AgentTrace()
    response = await client.responses.parse(
        model=model,
        reasoning={"effort": effort, "summary": "auto"},
        instructions=SYSTEM_PROMPT,
        input=[{"role": "user", "content": transcript}],
        tools=TOOL_SCHEMAS,
        text_format=AgentEstimate,
        store=True,
    )

    for iteration in range(1, max_iterations + 1):
        calls = [
            item
            for item in (response.output or [])
            if getattr(item, "type", None) == "function_call"
        ]
        reasoning = _reasoning_summary(response)

        if not calls:
            estimate = response.output_parsed
            if estimate is None:
                # Model stopped without tools but also without a parseable
                # estimate — force one last structured turn.
                log.warning("agent_missing_final_estimate", iteration=iteration)
                final = await client.responses.parse(
                    model=model,
                    previous_response_id=response.id,
                    reasoning={"effort": effort, "summary": "auto"},
                    instructions=SYSTEM_PROMPT
                    + "\nReturn the final AgentEstimate now. No more tools.",
                    input=[
                        {
                            "role": "user",
                            "content": "Produce the final structured estimate now.",
                        }
                    ],
                    text_format=AgentEstimate,
                    store=True,
                )
                estimate = final.output_parsed
                if estimate is None:
                    raise RuntimeError("Agent finished without a structured estimate.")
                trace.steps.append(
                    AgentStep(
                        reasoning=_reasoning_summary(final),
                        action="(final structured output)",
                        observation=estimate.model_dump_json(),
                    )
                )
            log.info(
                "agent_finished",
                iterations=iteration - 1,
                components=len(estimate.components),
                total_hours=estimate.total_hours,
            )
            return estimate, trace

        tool_outputs: list[dict[str, Any]] = []
        for call in calls:
            args = json.loads(call.arguments)
            observation = await dispatch_tool(
                call.name,
                args,
                retrieval_backend=retrieval_backend,
            )
            trace.steps.append(
                AgentStep(
                    reasoning=reasoning,
                    action=_format_action(call.name, call.arguments),
                    observation=_observation_summary(observation),
                )
            )
            tool_outputs.append(
                {
                    "type": "function_call_output",
                    "call_id": call.call_id,
                    "output": observation,
                }
            )
            # Only attach the full reasoning summary to the first call of the turn.
            reasoning = "(same turn)"

        log.info(
            "agent_turn",
            iteration=iteration,
            n_calls=len(calls),
            tools=[c.name for c in calls],
        )

        response = await client.responses.parse(
            model=model,
            previous_response_id=response.id,
            reasoning={"effort": effort, "summary": "auto"},
            instructions=SYSTEM_PROMPT,
            input=tool_outputs,
            tools=TOOL_SCHEMAS,
            text_format=AgentEstimate,
            store=True,
        )

    raise RuntimeError(
        f"Agent exceeded max_iterations={max_iterations} without finishing."
    )
