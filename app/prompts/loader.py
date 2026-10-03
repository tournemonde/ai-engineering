"""Render versioned Jinja prompt templates for the estimator."""

from __future__ import annotations

from pathlib import Path

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from app.context.examples import format_examples_for_prompt, select_examples
from app.schemas.estimation import EstimationRequest

PROMPTS_DIR = Path(__file__).parent

# snippet: fail on missing variables instead of rendering them as empty strings
_env = Environment(
    loader=FileSystemLoader(PROMPTS_DIR),
    trim_blocks=True,
    lstrip_blocks=True,
    keep_trailing_newline=False,
    undefined=StrictUndefined,
)


def render_estimation_prompt(
    request: EstimationRequest,
    version: str = "v1",
) -> tuple[str, str]:
    """Return (system, user) text for one estimation call.

    ``version`` selects ``estimation/<version>/`` so a later v2 can be rendered
    without changing callers.
    """
    system = _env.get_template(f"estimation/{version}/system.j2")
    user = _env.get_template(f"estimation/{version}/user.j2")

    reference_examples = ""
    if request.use_examples and request.num_examples > 0:
        reference_examples = format_examples_for_prompt(
            select_examples(request.num_examples),
            request.example_format,
        )

    context = {
        "description": request.transcription,
        "project_type": request.project_type.value,
        "detail_level": request.detail_level.value,
        "output_format": request.output_format.value,
        "use_examples": request.use_examples,
        "inline_cleaning": request.preprocessing == "inline_cleaning",
        "reference_examples": reference_examples,
    }

    return system.render(**context), user.render(**context)
