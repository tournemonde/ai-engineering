"""Estimation orchestration: prompt building, optional preprocessing, and dispatch
to the LLM. The actual provider calls now live in :mod:`app.services.llm_wrapper`,
so this module focuses on Session 2 concerns (knobs, prompt assembly) while the
wrapper handles cache, fallback, and cost tracking transparently.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import structlog

from app.dependencies import get_llm_wrapper
from app.prompts.loader import render_estimation_prompt
from app.schemas.estimation import (
    DetailLevel,
    EstimationRequest,
    ExampleFormat,
    OutputFormat,
    PreprocessingMode,
    ProjectType,
)

log = structlog.get_logger()

DEFAULT_MAX_TOKENS = 4000
EXTRACTION_MAX_TOKENS = 1500


class LLMServiceError(Exception):
    """Raised when the LLM provider call fails."""


# ---------------------------------------------------------------------------
# Prompt building blocks
#
# The two ACTIVE_OUTPUT_PROMPT variants live side by side so the instructor
# can switch between them in the live session (Block 3.4) by editing the
# ACTIVE_OUTPUT_PROMPT assignment below. Uvicorn `--reload` picks up the
# change automatically.
# ---------------------------------------------------------------------------

PROMPT_OUTPUT_BASIC = "Generate an estimation for the project described above."

PROMPT_OUTPUT_STRUCTURED = """\
Generate the estimation with this exact structure:

## Project summary
[2-3 sentences describing the project scope and goals]

## Task breakdown
| Task | Hours | Cost (EUR) |
[one row per task; cost = hours * 62.50 EUR for developer tasks]

## Totals
- Total hours: [number]
- Total cost: [number] EUR
- Recommended team: [composition]
- Estimated duration: [weeks]

## Risks and assumptions
- [3-5 bullet points covering technical risks, scope assumptions, and external dependencies]
"""

# >>> Block 3.4 live switch: change the right-hand side to PROMPT_OUTPUT_STRUCTURED
ACTIVE_OUTPUT_PROMPT = PROMPT_OUTPUT_BASIC


INLINE_CLEANING_BLOCK = """\
The transcription you receive is from a real meeting and may contain:
- Informal small talk you must ignore
- Implicit requirements you must surface explicitly
- Contradictions where you must trust the most recent statement
- Non-technical jargon you must interpret

Extract ONLY the functional and technical requirements relevant to the estimation."""


EXTRACTION_SYSTEM_PROMPT = (
    "You are an analyst. Read the meeting transcription and produce a clean, "
    "deduplicated bullet list of functional requirements, non-functional "
    "requirements, integrations, constraints and explicit deadlines. Ignore "
    "fillers, divagations and off-topic remarks. Output Markdown only."
)


@dataclass
class GenerationOptions:
    """Per-request knobs that drive prompt construction and the LLM call."""

    preprocessing: PreprocessingMode = "none"
    example_format: ExampleFormat = "markdown"
    num_examples: int = 3
    use_examples: bool = True
    model: str | None = None
    max_tokens: int = DEFAULT_MAX_TOKENS
    thinking_budget: int | None = None
    project_type: ProjectType = ProjectType.WEB_SAAS
    detail_level: DetailLevel = DetailLevel.MEDIUM
    output_format: OutputFormat = OutputFormat.PHASES_TABLE


# ---------------------------------------------------------------------------
# System prompt construction
# ---------------------------------------------------------------------------


def build_system_prompt(
    example_format: ExampleFormat = "markdown",
    num_examples: int = 3,
    use_examples: bool = True,
    inline_cleaning: bool = False,
    *,
    description: str = "x" * 50,
    project_type: ProjectType = ProjectType.WEB_SAAS,
    detail_level: DetailLevel = DetailLevel.MEDIUM,
    output_format: OutputFormat = OutputFormat.PHASES_TABLE,
) -> str:
    """Assemble the system prompt from the v1 Jinja template.

    Kept as a function so the streaming route and older callers share one entry.
    The live switch ``ACTIVE_OUTPUT_PROMPT`` is still appended at the end.
    """
    request = EstimationRequest.model_construct(
        transcription=description,
        preprocessing="inline_cleaning" if inline_cleaning else "none",
        example_format=example_format,
        num_examples=num_examples,
        use_examples=use_examples,
        project_type=project_type,
        detail_level=detail_level,
        output_format=output_format,
    )
    system, _user = render_estimation_prompt(request)
    return f"{system}\n\n{ACTIVE_OUTPUT_PROMPT}"


# ---------------------------------------------------------------------------
# LLM dispatch (single seam — tests monkeypatch this)
# ---------------------------------------------------------------------------


def _invoke_llm(
    *,
    system_prompt: str,
    user_message: str,
    model_override: str | None,
    max_tokens: int,
    thinking_budget: int | None,
) -> dict[str, Any]:
    """Single seam through which every LLM call passes. Tests monkeypatch this."""
    wrapper = get_llm_wrapper()
    return wrapper.complete(
        system_prompt=system_prompt,
        user_message=user_message,
        model_override=model_override,
        max_tokens=max_tokens,
        thinking_budget=thinking_budget,
    )


# ---------------------------------------------------------------------------
# Two-phase preprocessing (phase 1: requirement extraction)
# ---------------------------------------------------------------------------


def extract_requirements(
    transcription: str,
    opts: GenerationOptions,
) -> tuple[str, dict, float]:
    """Run the cheap phase-1 LLM call that turns a raw transcription into clean requirements.

    Returns ``(requirements_text, usage_dict, cost_usd)``.
    """
    log.info("extracting_requirements", model_override=opts.model)

    result = _invoke_llm(
        system_prompt=EXTRACTION_SYSTEM_PROMPT,
        user_message=transcription,
        model_override=opts.model,
        max_tokens=EXTRACTION_MAX_TOKENS,
        thinking_budget=None,
    )

    return (
        result["estimation"],
        {
            "input": result["usage"]["input_tokens"],
            "output": result["usage"]["output_tokens"],
        },
        float(result.get("cost_usd", 0.0)),
    )


# ---------------------------------------------------------------------------
# Main entrypoint
# ---------------------------------------------------------------------------


def generate_estimation(
    transcription: str,
    opts: GenerationOptions | None = None,
) -> dict[str, Any]:
    """Generate a software estimation from a meeting transcription using the configured LLM."""
    opts = opts or GenerationOptions()

    t0 = time.perf_counter()

    prep_usage = {"input": 0, "output": 0}
    prep_cost = 0.0
    extracted_requirements: str | None = None
    user_input = transcription

    if opts.preprocessing == "two_phase":
        extracted_requirements, prep_usage, prep_cost = extract_requirements(transcription, opts)
        user_input = extracted_requirements

    # snippet: product prompt (Jinja) plus the instructor live-switch paragraph
    prompt_request = EstimationRequest.model_construct(
        transcription=user_input,
        preprocessing=opts.preprocessing,
        example_format=opts.example_format,
        num_examples=opts.num_examples,
        use_examples=opts.use_examples,
        model=opts.model,
        max_tokens=opts.max_tokens,
        thinking_budget=opts.thinking_budget,
        project_type=opts.project_type,
        detail_level=opts.detail_level,
        output_format=opts.output_format,
    )
    system_prompt, user_message = render_estimation_prompt(prompt_request)
    system_prompt = f"{system_prompt}\n\n{ACTIVE_OUTPUT_PROMPT}"

    log.info(
        "generating_estimation",
        model_override=opts.model,
        preprocessing=opts.preprocessing,
        example_format=opts.example_format,
        num_examples=opts.num_examples,
        use_examples=opts.use_examples,
        max_tokens=opts.max_tokens,
        thinking_budget=opts.thinking_budget,
    )

    try:
        result = _invoke_llm(
            system_prompt=system_prompt,
            user_message=user_message,
            model_override=opts.model,
            max_tokens=opts.max_tokens,
            thinking_budget=opts.thinking_budget,
        )
    except Exception as exc:
        log.error("llm_call_failed", error=str(exc), error_type=type(exc).__name__)
        raise LLMServiceError(f"LLM call failed: {exc}") from exc

    result["usage"]["preprocessing_input_tokens"] = prep_usage["input"]
    result["usage"]["preprocessing_output_tokens"] = prep_usage["output"]
    result["preprocessing"] = opts.preprocessing
    result["extracted_requirements"] = extracted_requirements
    result["latency_ms"] = int((time.perf_counter() - t0) * 1000)
    result["cost_usd"] = round(float(result.get("cost_usd", 0.0)) + prep_cost, 6)
    # ``cache_hit`` is whatever the wrapper returned for the main estimation call.
    result.setdefault("cache_hit", False)

    return result
