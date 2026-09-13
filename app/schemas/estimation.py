"""Pydantic contracts for estimation endpoints."""

from __future__ import annotations

from pydantic import BaseModel, Field


class EstimationRequest(BaseModel):
    """Incoming meeting transcription to estimate."""

    transcription: str = Field(
        ...,
        min_length=50,
        description="Transcripción de la reunión con el cliente",
    )


class EstimationResponse(BaseModel):
    """LLM-generated software estimation response."""

    estimation: str
    model: str
    provider: str
    input_tokens: int | None = None
    output_tokens: int | None = None
