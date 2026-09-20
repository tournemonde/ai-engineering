"""Estimate endpoint tests with mocked LLM calls."""

from __future__ import annotations

from typing import Any

import pytest
from fastapi.testclient import TestClient

from app.config import Settings, get_settings
from app.main import app

SAMPLE_TRANSCRIPTION = (
    "En la reunión con el equipo de marketing, el cliente explicó que necesita una "
    "landing page con formulario de contacto, integración con su CRM actual (HubSpot), "
    "y una sección de blog con editor WYSIWYG. El plazo ideal sería tenerlo listo en "
    "4 semanas. El diseño ya existe en Figma."
)


@pytest.fixture
def openai_settings() -> Settings:
    return Settings(
        LLM_PROVIDER="openai",
        LLM_MODEL="gpt-4o-mini",
        OPENAI_API_KEY="test-openai-key",
        ANTHROPIC_API_KEY=None,
    )


@pytest.fixture
def client(openai_settings: Settings) -> TestClient:
    app.dependency_overrides[get_settings] = lambda: openai_settings
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


def test_estimate_validation_rejects_short_transcription(client: TestClient) -> None:
    response = client.post("/api/v1/estimate", json={"transcription": "too short"})
    assert response.status_code == 422


def test_estimate_success_with_mocked_llm(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_generate(transcription: str, settings: Settings | None = None) -> dict[str, Any]:
        assert "HubSpot" in transcription
        return {
            "estimation": "## Estimación: Landing page\n\n**Total estimado: 80 horas**",
            "model": "gpt-4o-mini",
            "provider": "openai",
            "input_tokens": 100,
            "output_tokens": 50,
        }

    monkeypatch.setattr("app.routers.estimations.generate_estimation", fake_generate)
    response = client.post(
        "/api/v1/estimate",
        json={"transcription": SAMPLE_TRANSCRIPTION},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["provider"] == "openai"
    assert payload["model"] == "gpt-4o-mini"
    assert "Estimación" in payload["estimation"]
    assert payload["input_tokens"] == 100
    assert payload["output_tokens"] == 50


def test_estimate_maps_provider_errors_to_502(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def boom(transcription: str, settings: Settings | None = None) -> dict[str, Any]:
        raise RuntimeError("provider down")

    monkeypatch.setattr("app.routers.estimations.generate_estimation", boom)
    response = client.post(
        "/api/v1/estimate",
        json={"transcription": SAMPLE_TRANSCRIPTION},
    )
    assert response.status_code == 502
    detail = response.json()["detail"]
    assert detail["error"] == "llm_provider_error"
