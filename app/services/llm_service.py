"""LLM service for CAG software estimations (async + sync streaming)."""

from __future__ import annotations

import time
from collections.abc import Iterator
from typing import Any, TypedDict

from anthropic import Anthropic, AsyncAnthropic
from openai import AsyncOpenAI, OpenAI

from app.config import Settings, get_settings
from app.context.examples import ESTIMATION_EXAMPLES, EstimationExample


class StreamMetrics(TypedDict, total=False):
    """Side-channel metrics filled while streaming tokens."""

    model: str
    provider: str
    input_tokens: int | None
    output_tokens: int | None
    latency_ms: float


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
Úsalas como referencia para calibrar tus estimaciones: las tarifas, la
granularidad del desglose de tareas y la estructura del presupuesto deben
ser consistentes con estos ejemplos.

Tarifas de la empresa (jornada de 8 horas):
- Desarrollo: 500 EUR/día (62,50 EUR/hora)
- Diseño UX: 400 EUR/día (50 EUR/hora)

Tu salida DEBE seguir este formato exacto:
- Título del proyecto como heading H2 (## Estimación: ...)
- Resumen del proyecto (2-3 frases)
- Tabla de desglose con columnas: Tarea, Horas, Coste (EUR)
- Totales: horas y coste en EUR
- Equipo recomendado
- Duración estimada en semanas

Alinea las horas a medios días o días (8, 12, 16, 24, ...). Moneda EUR.
Responde en Markdown con la misma estructura que los ejemplos de referencia.

{examples_text}
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


# snippet: sync OpenAI token stream
def _stream_openai(
    *,
    settings: Settings,
    system_prompt: str,
    transcription: str,
    metrics: StreamMetrics,
) -> Iterator[str]:
    client = OpenAI(api_key=settings.OPENAI_API_KEY)
    start = time.perf_counter()
    stream = client.chat.completions.create(
        model=settings.LLM_MODEL,
        max_tokens=settings.MAX_OUTPUT_TOKENS,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": transcription},
        ],
        stream=True,
        stream_options={"include_usage": True},
    )
    for chunk in stream:
        if chunk.choices:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta
        usage = getattr(chunk, "usage", None)
        if usage is not None:
            metrics["input_tokens"] = usage.prompt_tokens
            metrics["output_tokens"] = usage.completion_tokens
    metrics["model"] = settings.LLM_MODEL
    metrics["provider"] = "openai"
    metrics["latency_ms"] = round((time.perf_counter() - start) * 1000, 1)


# snippet: sync Anthropic token stream
def _stream_anthropic(
    *,
    settings: Settings,
    system_prompt: str,
    transcription: str,
    metrics: StreamMetrics,
) -> Iterator[str]:
    client = Anthropic(api_key=settings.ANTHROPIC_API_KEY)
    start = time.perf_counter()
    with client.messages.stream(
        model=settings.LLM_MODEL,
        max_tokens=settings.MAX_OUTPUT_TOKENS,
        system=system_prompt,
        messages=[{"role": "user", "content": transcription}],
    ) as stream:
        for text in stream.text_stream:
            if text:
                yield text
        final = stream.get_final_message()
        usage = final.usage
        metrics["input_tokens"] = usage.input_tokens if usage else None
        metrics["output_tokens"] = usage.output_tokens if usage else None
    metrics["model"] = settings.LLM_MODEL
    metrics["provider"] = "anthropic"
    metrics["latency_ms"] = round((time.perf_counter() - start) * 1000, 1)


def stream_estimation(
    transcription: str,
    settings: Settings | None = None,
    metrics: StreamMetrics | None = None,
) -> Iterator[str]:
    """Yield estimation tokens for Streamlit (single-turn CAG, sync SDKs)."""
    resolved = settings or get_settings()
    system_prompt = build_system_prompt()
    side_channel: StreamMetrics = metrics if metrics is not None else {}

    if resolved.LLM_PROVIDER == "openai":
        yield from _stream_openai(
            settings=resolved,
            system_prompt=system_prompt,
            transcription=transcription,
            metrics=side_channel,
        )
        return
    yield from _stream_anthropic(
        settings=resolved,
        system_prompt=system_prompt,
        transcription=transcription,
        metrics=side_channel,
    )
