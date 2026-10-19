"""Session 7 embedding pipeline — chunker and ingest endpoint (no live API)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.dependencies import get_embedder
from app.embedding_pipeline.embedder import estimated_cost_usd
from app.embedding_pipeline.schemas import Chunk, EmbeddedChunk
from app.main import app

_SAMPLE = Path(__file__).resolve().parents[1] / "data" / "budgets_sample.json"


class FakeEmbedder:
    def embed_one(self, text: str) -> list[float]:
        return [1.0, 0.0, 0.0]

    def embed_many(self, chunks: list[Chunk]) -> list[EmbeddedChunk]:
        return [
            EmbeddedChunk(
                chunk_id=c.chunk_id,
                text=c.text,
                metadata=c.metadata,
                token_count=c.token_count,
                embedding=[1.0, 0.0, 0.0],
            )
            for c in chunks
        ]


@pytest.fixture
def sample_budgets() -> list[dict]:
    return json.loads(_SAMPLE.read_text(encoding="utf-8"))


def test_structural_chunker_one_component_per_chunk(sample_budgets: list[dict]) -> None:
    from app.embedding_pipeline.chunker import JSONStructuralChunker

    chunker = JSONStructuralChunker()
    budgets_payload = {"budgets": sample_budgets[:1]}
    from app.embedding_pipeline.schemas import IngestRequest

    request = IngestRequest.model_validate(budgets_payload)
    chunks = chunker.chunk(request.budgets)
    assert len(chunks) == len(request.budgets[0].components)
    assert chunks[0].chunk_id.startswith("BUD-2024-001::")
    assert "OAuth" in chunks[0].text
    assert chunks[0].token_count > 0
    assert chunks[0].metadata["client_sector"] == "finance"


def test_estimated_cost_usd_scales_with_tokens() -> None:
    assert estimated_cost_usd(1_000_000) == pytest.approx(0.02)
    assert estimated_cost_usd(0) == 0.0


def test_ingest_endpoint_with_fake_embedder(
    client: TestClient,
    sample_budgets: list[dict],
) -> None:
    app.dependency_overrides[get_embedder] = lambda: FakeEmbedder()
    try:
        response = client.post(
            "/embeddings/ingest",
            json={"budgets": sample_budgets[:2]},
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    body = response.json()
    assert body["stats"]["total_budgets"] == 2
    assert body["stats"]["total_chunks"] == len(body["chunks"])
    assert body["stats"]["total_tokens"] > 0
    assert len(body["chunks"][0]["embedding"]) == 3


def test_ingest_without_api_key_returns_500(
    client: TestClient, sample_budgets: list[dict]
) -> None:
    app.dependency_overrides[get_embedder] = lambda: None
    try:
        response = client.post(
            "/embeddings/ingest",
            json={"budgets": sample_budgets[:1]},
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 500
    assert response.json()["detail"] == "Embedding service is not available."
