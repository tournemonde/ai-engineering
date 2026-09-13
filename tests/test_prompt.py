"""CAG prompt construction tests."""

from __future__ import annotations

from app.context.examples import ESTIMATION_EXAMPLES
from app.services.llm_service import build_system_prompt, format_examples


def test_at_least_two_static_examples() -> None:
    assert len(ESTIMATION_EXAMPLES) >= 2


def test_format_examples_uses_delimiters() -> None:
    text = format_examples(ESTIMATION_EXAMPLES)
    assert "===== ESTIMACIÓN DE REFERENCIA 1 =====" in text
    assert "===== FIN DE ESTIMACIONES DE REFERENCIA =====" in text


def test_system_prompt_includes_examples_and_rules() -> None:
    prompt = build_system_prompt()
    assert "consultor senior" in prompt.lower() or "experiencia" in prompt.lower()
    assert "EUR" in prompt
    assert "múltiplos de 5" in prompt or "multiplos de 5" in prompt
    assert ESTIMATION_EXAMPLES[0]["meeting_summary"][:40] in prompt
