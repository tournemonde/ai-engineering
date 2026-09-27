import asyncio
from collections.abc import AsyncIterator

import structlog
from fastapi import APIRouter, Depends, HTTPException
from sse_starlette.sse import EventSourceResponse

from app.dependencies import get_llm_wrapper
from app.prompts.loader import render_estimation_prompt
from app.schemas.estimation import (
    EstimationRequest,
    EstimationResponse,
    StreamEstimationRequest,
)
from app.services.evaluation import evaluate_estimation_structure
from app.services.llm_service import (
    ACTIVE_OUTPUT_PROMPT,
    GenerationOptions,
    LLMServiceError,
    generate_estimation,
)
from app.services.llm_wrapper import LLMWrapper

log = structlog.get_logger()

router = APIRouter(prefix="/api/v1", tags=["estimations"])


@router.post("/estimate", response_model=EstimationResponse)
async def create_estimation(request: EstimationRequest) -> EstimationResponse:
    """Receive a meeting transcription and return a software project estimation."""
    opts = GenerationOptions(
        preprocessing=request.preprocessing,
        example_format=request.example_format,
        num_examples=request.num_examples,
        use_examples=request.use_examples,
        model=request.model,
        max_tokens=request.max_tokens,
        thinking_budget=request.thinking_budget,
        project_type=request.project_type,
        detail_level=request.detail_level,
        output_format=request.output_format,
    )

    try:
        result = generate_estimation(request.transcription, opts)
    except LLMServiceError as exc:
        log.error("estimation_endpoint_error", error=str(exc))
        raise HTTPException(status_code=500, detail=str(exc))

    validation = (
        evaluate_estimation_structure(result["estimation"], result["finish_reason"])
        if request.evaluate
        else None
    )

    return EstimationResponse(**result, validation=validation)


@router.post("/estimate/stream")
async def create_estimation_stream(
    request: StreamEstimationRequest,
    wrapper: LLMWrapper = Depends(get_llm_wrapper),
) -> EventSourceResponse:
    """Stream a software estimation token by token via Server-Sent Events.

    The streaming path is intentionally simpler than POST /estimate: it skips
    two-phase preprocessing and structural validation, since both fight the UX
    benefit of streaming (intermediate phase 1 tokens would leak; validation
    only makes sense over the complete text).
    """
    prompt_request = EstimationRequest(
        transcription=request.transcription,
        project_type=request.project_type,
        detail_level=request.detail_level,
        output_format=request.output_format,
        model=request.model,
        max_tokens=request.max_tokens,
    )
    system_prompt, user_message = render_estimation_prompt(prompt_request)
    system_prompt = f"{system_prompt}\n\n{ACTIVE_OUTPUT_PROMPT}"

    async def event_generator() -> AsyncIterator[dict]:
        loop = asyncio.get_running_loop()
        chunks = wrapper.complete_stream(
            system_prompt=system_prompt,
            user_message=user_message,
            model_override=request.model,
            max_tokens=request.max_tokens,
        )

        def _next_chunk() -> str | None:
            try:
                return next(chunks)
            except StopIteration:
                return None
            except Exception as exc:  # noqa: BLE001 — surface as SSE error event
                log.error("estimate_stream_failed", error=str(exc), error_type=type(exc).__name__)
                raise

        try:
            while True:
                chunk = await loop.run_in_executor(None, _next_chunk)
                if chunk is None:
                    break
                if chunk:
                    yield {"event": "token", "data": chunk}
            yield {"event": "done", "data": "[DONE]"}
        except Exception as exc:  # noqa: BLE001
            yield {"event": "error", "data": str(exc)}

    return EventSourceResponse(event_generator())
