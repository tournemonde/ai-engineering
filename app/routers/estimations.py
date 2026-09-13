"""Estimation HTTP endpoints."""

from __future__ import annotations

import logging
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException

from app.config import Settings, get_settings
from app.schemas.estimation import EstimationRequest, EstimationResponse
from app.services.llm_service import generate_estimation

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["estimations"])

SettingsDep = Annotated[Settings, Depends(get_settings)]


@router.post("/estimate", response_model=EstimationResponse)
async def estimate(
    request: EstimationRequest,
    settings: SettingsDep,
) -> EstimationResponse:
    """Generate a software estimation from a meeting transcription."""
    try:
        result = await generate_estimation(request.transcription, settings=settings)
    except Exception as exc:  # noqa: BLE001 — surface provider failures as 502
        logger.exception("LLM estimation failed")
        raise HTTPException(
            status_code=502,
            detail={"error": "llm_provider_error", "message": str(exc)},
        ) from exc

    return EstimationResponse.model_validate(result)
