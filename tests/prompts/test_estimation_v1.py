"""Template tests: they check rendered prompt text, not the LLM."""

from app.prompts.loader import render_estimation_prompt
from app.schemas.estimation import DetailLevel, EstimationRequest, OutputFormat, ProjectType


def _request(**overrides: object) -> EstimationRequest:
    payload: dict[str, object] = {
        "transcription": "Mobile app with login, chat and push notifications.",
        "project_type": ProjectType.MOBILE_APP,
        "detail_level": DetailLevel.DETAILED,
        "output_format": OutputFormat.PHASES_TABLE,
    }
    payload.update(overrides)
    return EstimationRequest.model_validate(payload)


def test_user_block_contains_description() -> None:
    _system, user = render_estimation_prompt(_request())

    assert "<project_description>" in user
    assert "Mobile app with login" in user


def test_phases_table_mentions_confidence_and_narrative_does_not() -> None:
    phases_system, _user = render_estimation_prompt(
        _request(output_format=OutputFormat.PHASES_TABLE, detail_level=DetailLevel.MEDIUM)
    )
    narrative_system, _user = render_estimation_prompt(
        _request(output_format=OutputFormat.NARRATIVE, detail_level=DetailLevel.MEDIUM)
    )

    assert "confidence_pct" in phases_system
    assert "confidence_pct" not in narrative_system


def test_detailed_asks_for_assumptions_and_summary_does_not() -> None:
    detailed_system, _user = render_estimation_prompt(_request(detail_level=DetailLevel.DETAILED))
    summary_system, _user = render_estimation_prompt(_request(detail_level=DetailLevel.SUMMARY))

    assert "list the assumptions" in detailed_system
    assert "list the assumptions" not in summary_system
