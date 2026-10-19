#!/usr/bin/env python3
"""Compare cosine similarity between two texts using OpenAIEmbedder."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

# Allow `python scripts/compare.py` from estimator/ without installing the package.
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from app.config import get_settings  # noqa: E402
from app.dependencies import get_openai_client  # noqa: E402
from app.embedding_pipeline.embedder import OpenAIEmbedder  # noqa: E402


def cosine_similarity(a: list[float], b: list[float]) -> float:
    if len(a) != len(b):
        raise ValueError("Vectors must have the same dimension")
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cosine similarity between two embedded texts"
    )
    parser.add_argument("--text-a", required=True, help="First text")
    parser.add_argument("--text-b", required=True, help="Second text")
    args = parser.parse_args()

    get_settings()
    client = get_openai_client()
    if client is None:
        print("OPENAI_API_KEY is not configured.", file=sys.stderr)
        sys.exit(1)

    embedder = OpenAIEmbedder(client)
    vec_a = embedder.embed_one(args.text_a)
    vec_b = embedder.embed_one(args.text_b)
    score = cosine_similarity(vec_a, vec_b)

    print(f"Text A: {args.text_a}")
    print(f"Text B: {args.text_b}")
    print(f"Cosine similarity: {score:.4f}")


if __name__ == "__main__":
    main()
