#!/usr/bin/env python3
"""Run Session 8 semantic-search examples against POST /search."""

from __future__ import annotations

import argparse
import textwrap

import httpx

from app.config import get_settings

QUERIES = [
    "REST API development with JWT authentication for financial sector",
    "secure backend service with token-based access control for banking applications",
    "mobile application for restaurant reservations",
    "integration with external system",
    "migration from monolith to microservices architecture using Kubernetes",
]


def render_result(result: dict) -> str:
    snippet = textwrap.shorten(result["content"], width=120, placeholder="...")
    return (
        f"chunk_id={result['chunk_id']} "
        f"distance={result['distance']:.4f} "
        f"chunk_type={result['chunk_type']} "
        f"content={snippet}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run five Session 8 search examples")
    parser.add_argument(
        "--base-url",
        default=get_settings().ESTIMATOR_API_BASE_URL,
        help="API base URL (default from ESTIMATOR_API_BASE_URL)",
    )
    parser.add_argument("--k", type=int, default=5, help="Top-k results")
    args = parser.parse_args()

    base_url = args.base_url.rstrip("/")
    with httpx.Client(timeout=60.0) as client:
        for idx, query in enumerate(QUERIES, start=1):
            response = client.post(f"{base_url}/search", json={"query": query, "k": args.k})
            response.raise_for_status()
            payload = response.json()
            print(f"\n=== Query {idx}: {query}")
            print(f"search_time_ms={payload['search_time_ms']} k={payload['k']}")
            for row in payload["results"]:
                print(render_result(row))


if __name__ == "__main__":
    main()
