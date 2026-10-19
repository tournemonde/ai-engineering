"""Pydantic models for the embedding ingest pipeline."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

ClientSector = Literal["finance", "ecommerce", "healthcare", "industrial"]
ComponentComplexity = Literal["low", "medium", "high"]


class ClientMetadata(BaseModel):
    name: str
    sector: ClientSector
    country: str = Field(min_length=2, max_length=2)


class BudgetComponent(BaseModel):
    component_id: str
    name: str
    description: str
    tech_stack: list[str]
    estimated_hours: int = Field(ge=0)
    complexity: ComponentComplexity
    dependencies: list[str] = Field(default_factory=list)


class Budget(BaseModel):
    budget_id: str
    client_metadata: ClientMetadata
    project_summary: str
    main_technology: str
    year: int = Field(ge=2000, le=2100)
    total_estimated_hours: int = Field(ge=0)
    components: list[BudgetComponent] = Field(min_length=1)


class Chunk(BaseModel):
    chunk_id: str
    text: str
    metadata: dict[str, Any]
    token_count: int = Field(ge=0)


class EmbeddedChunk(Chunk):
    embedding: list[float]


class IngestRequest(BaseModel):
    budgets: list[Budget] = Field(min_length=1)


class IngestStats(BaseModel):
    total_budgets: int
    total_chunks: int
    total_tokens: int
    estimated_cost_usd: float


class IngestResponse(BaseModel):
    chunks: list[EmbeddedChunk]
    stats: IngestStats
