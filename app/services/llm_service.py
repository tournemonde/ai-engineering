"""Async LLM service for CAG software estimations."""

from __future__ import annotations

from typing import Any

from anthropic import AsyncAnthropic
from openai import AsyncOpenAI

from app.config import Settings, get_settings
from app.context.examples import ESTIMATION_EXAMPLES, EstimationExample


def format_examples(examples: list[EstimationExample]) -> str:
    """Format static examples as delimited Markdown for the system prompt."""
    blocks: list[str] = []
    for index, example in enumerate(examples, start=1):
        blocks.append(
            "\n".join(
                [
                    f"===== ESTIMACIÓN DE REFERENCIA {index} =====",
                    "### Resumen de la reunión original",
                    example["meeting_summary"].strip(),
                    "",
                    "### Estimación generada",
                    example["estimation"].strip(),
                    "",
                ]
            )
        )
    blocks.append("===== FIN DE ESTIMACIONES DE REFERENCIA =====")
    return "\n".join(blocks)


def build_system_prompt(examples: list[EstimationExample] | None = None) -> str:
    """Build the CAG system prompt with role, rules, and reference examples."""
    selected = examples if examples is not None else ESTIMATION_EXAMPLES
    examples_text = format_examples(selected)
    return f"""Eres un consultor senior de software con 15 años de experiencia en estimación
de proyectos. Tu trabajo es analizar transcripciones de reuniones con clientes
y generar estimaciones detalladas de desarrollo de software.

A continuación se incluyen estimaciones de proyectos anteriores de la empresa.
Úsalas como referencia para calibrar tus estimaciones: los precios por hora,
la granularidad del desglose de tareas y la estructura del presupuesto deben
ser consistentes con estos ejemplos.

{examples_text}

Tu estimación debe incluir:
1. Resumen del proyecto (2-3 frases)
2. Desglose de tareas con horas estimadas y coste
3. Equipo recomendado
4. Duración total estimada
5. Riesgos o supuestos clave

Usa EUR como moneda. Redondea las horas a múltiplos de 5.
Responde en Markdown con la misma estructura que los ejemplos de referencia.
"""


async def _call_openai(
    *,
    settings: Settings,
    system_prompt: str,
    transcription: str,
) -> dict[str, Any]:
    client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
    response = await client.chat.completions.create(
        model=settings.LLM_MODEL,
        max_tokens=settings.MAX_OUTPUT_TOKENS,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": transcription},
        ],
    )
    content = response.choices[0].message.content or ""
    usage = response.usage
    return {
        "estimation": content,
        "model": settings.LLM_MODEL,
        "provider": "openai",
        "input_tokens": usage.prompt_tokens if usage else None,
        "output_tokens": usage.completion_tokens if usage else None,
    }


async def _call_anthropic(
    *,
    settings: Settings,
    system_prompt: str,
    transcription: str,
) -> dict[str, Any]:
    client = AsyncAnthropic(api_key=settings.ANTHROPIC_API_KEY)
    response = await client.messages.create(
        model=settings.LLM_MODEL,
        max_tokens=settings.MAX_OUTPUT_TOKENS,
        system=system_prompt,
        messages=[{"role": "user", "content": transcription}],
    )
    text_parts = [
        block.text for block in response.content if getattr(block, "type", None) == "text"
    ]
    usage = response.usage
    return {
        "estimation": "\n".join(text_parts),
        "model": settings.LLM_MODEL,
        "provider": "anthropic",
        "input_tokens": usage.input_tokens if usage else None,
        "output_tokens": usage.output_tokens if usage else None,
    }


async def generate_estimation(
    transcription: str,
    settings: Settings | None = None,
) -> dict[str, Any]:
    """Generate a software estimation from a meeting transcription (single-turn CAG)."""
    resolved = settings or get_settings()
    system_prompt = build_system_prompt()

    if resolved.LLM_PROVIDER == "openai":
        return await _call_openai(
            settings=resolved,
            system_prompt=system_prompt,
            transcription=transcription,
        )
    return await _call_anthropic(
        settings=resolved,
        system_prompt=system_prompt,
        transcription=transcription,
    )
