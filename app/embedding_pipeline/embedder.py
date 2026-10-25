"""OpenAI embedding client with batching and simple rate-limit retries."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import structlog
from openai import OpenAI, RateLimitError

from app.embedding_pipeline.schemas import Chunk, EmbeddedChunk

if TYPE_CHECKING:
    from collections.abc import Sequence

log = structlog.get_logger()

# text-embedding-3-small input price (USD per 1M tokens) — update when OpenAI changes pricing.
_EMBEDDING_USD_PER_MILLION_TOKENS = 0.02
_DEFAULT_BATCH_SIZE = 100
_RETRY_DELAYS_SEC = (1.0, 2.0, 4.0)


def estimated_cost_usd(total_tokens: int) -> float:
    """Return estimated embedding cost for the given token count."""
    return (total_tokens / 1_000_000) * _EMBEDDING_USD_PER_MILLION_TOKENS


class OpenAIEmbedder:
    """Embed chunk text via the OpenAI Embeddings API."""

    def __init__(
        self,
        client: OpenAI,
        *,
        model: str = "text-embedding-3-small",
        batch_size: int = _DEFAULT_BATCH_SIZE,
    ) -> None:
        self._client = client
        self._model = model
        self._batch_size = batch_size

    def embed_one(self, text: str) -> list[float]:
        response = self._create_embeddings([text])
        return list(response.data[0].embedding)

    def embed_many(self, chunks: list[Chunk]) -> list[EmbeddedChunk]:
        if not chunks:
            return []

        embedded: list[EmbeddedChunk] = []
        for start in range(0, len(chunks), self._batch_size):
            batch = chunks[start : start + self._batch_size]
            texts = [chunk.text for chunk in batch]
            batch_tokens = sum(chunk.token_count for chunk in batch)
            t0 = time.perf_counter()
            response = self._create_embeddings(texts)
            latency_ms = (time.perf_counter() - t0) * 1000
            log.info(
                "embedding_batch_done",
                batch_size=len(batch),
                total_tokens=batch_tokens,
                latency_ms=round(latency_ms, 2),
            )
            for chunk, item in zip(batch, response.data, strict=True):
                embedded.append(
                    EmbeddedChunk(
                        chunk_id=chunk.chunk_id,
                        text=chunk.text,
                        metadata=chunk.metadata,
                        token_count=chunk.token_count,
                        embedding=list(item.embedding),
                    )
                )
        return embedded

    def _create_embeddings(self, texts: Sequence[str]):
        attempts = len(_RETRY_DELAYS_SEC) + 1
        for attempt in range(attempts):
            try:
                return self._client.embeddings.create(
                    model=self._model, input=list(texts)
                )
            except RateLimitError:
                if attempt >= attempts - 1:
                    raise
                delay = _RETRY_DELAYS_SEC[attempt]
                log.warning(
                    "embedding_rate_limited",
                    attempt=attempt + 1,
                    delay_sec=delay,
                    batch_size=len(texts),
                )
                time.sleep(delay)
        raise RuntimeError("unreachable")
