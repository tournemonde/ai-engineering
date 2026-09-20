"""Streaming estimation tests with mocked sync LLM SDKs."""

from __future__ import annotations

from collections.abc import Iterator
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest

from app.config import Settings
from app.services.llm_service import StreamMetrics, stream_estimation


@pytest.fixture
def openai_settings() -> Settings:
    return Settings(
        LLM_PROVIDER="openai",
        LLM_MODEL="gpt-4o-mini",
        OPENAI_API_KEY="test-openai-key",
        ANTHROPIC_API_KEY=None,
    )


@pytest.fixture
def anthropic_settings() -> Settings:
    return Settings(
        LLM_PROVIDER="anthropic",
        LLM_MODEL="claude-haiku-4-5",
        OPENAI_API_KEY=None,
        ANTHROPIC_API_KEY="test-anthropic-key",
    )


def _openai_chunk(content: str | None, usage: Any = None) -> SimpleNamespace:
    choice = SimpleNamespace(delta=SimpleNamespace(content=content))
    return SimpleNamespace(choices=[choice], usage=usage)


def test_stream_estimation_openai_yields_tokens_and_metrics(
    openai_settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    usage = SimpleNamespace(prompt_tokens=120, completion_tokens=45)
    chunks = [
        _openai_chunk("## Estimación: "),
        _openai_chunk("Landing"),
        _openai_chunk(None, usage=usage),
    ]

    fake_client = MagicMock()
    fake_client.chat.completions.create.return_value = iter(chunks)

    monkeypatch.setattr(
        "app.services.llm_service.OpenAI",
        lambda **_kwargs: fake_client,
    )

    metrics: StreamMetrics = {}
    tokens = list(
        stream_estimation(
            "Transcripción de prueba con suficiente contenido para estimar.",
            settings=openai_settings,
            metrics=metrics,
        )
    )

    assert "".join(tokens) == "## Estimación: Landing"
    assert metrics["provider"] == "openai"
    assert metrics["model"] == "gpt-4o-mini"
    assert metrics["input_tokens"] == 120
    assert metrics["output_tokens"] == 45
    assert "latency_ms" in metrics
    fake_client.chat.completions.create.assert_called_once()
    call_kwargs = fake_client.chat.completions.create.call_args.kwargs
    assert call_kwargs["stream"] is True
    assert call_kwargs["stream_options"] == {"include_usage": True}
    assert call_kwargs["messages"][0]["role"] == "system"
    assert "consultor senior" in call_kwargs["messages"][0]["content"].lower()


def test_stream_estimation_anthropic_yields_tokens_and_metrics(
    anthropic_settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeStream:
        def __init__(self) -> None:
            self.text_stream: Iterator[str] = iter(["## Estimación: ", "App móvil"])

        def __enter__(self) -> FakeStream:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def get_final_message(self) -> SimpleNamespace:
            return SimpleNamespace(usage=SimpleNamespace(input_tokens=200, output_tokens=80))

    fake_messages = MagicMock()
    fake_messages.stream.return_value = FakeStream()
    fake_client = MagicMock()
    fake_client.messages = fake_messages

    monkeypatch.setattr(
        "app.services.llm_service.Anthropic",
        lambda **_kwargs: fake_client,
    )

    metrics: StreamMetrics = {}
    tokens = list(
        stream_estimation(
            "Transcripción de prueba con suficiente contenido para estimar.",
            settings=anthropic_settings,
            metrics=metrics,
        )
    )

    assert "".join(tokens) == "## Estimación: App móvil"
    assert metrics["provider"] == "anthropic"
    assert metrics["model"] == "claude-haiku-4-5"
    assert metrics["input_tokens"] == 200
    assert metrics["output_tokens"] == 80
    assert "latency_ms" in metrics
    fake_messages.stream.assert_called_once()
    call_kwargs = fake_messages.stream.call_args.kwargs
    assert call_kwargs["system"]
    assert "EUR" in call_kwargs["system"]
