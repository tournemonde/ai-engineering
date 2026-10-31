#!/usr/bin/env python3
"""Ingest data/budgets_sample.json into Session 8 /embeddings/ingest endpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import httpx

from app.config import get_settings


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest sample budgets for Session 8")
    parser.add_argument(
        "--base-url",
        default=get_settings().ESTIMATOR_API_BASE_URL,
        help="API base URL (default from ESTIMATOR_API_BASE_URL)",
    )
    parser.add_argument(
        "--input",
        default="data/budgets_sample.json",
        help="Path to budgets array JSON file",
    )
    args = parser.parse_args()

    budgets = json.loads(Path(args.input).read_text(encoding="utf-8"))
    base_url = args.base_url.rstrip("/")
    success = 0
    duplicate = 0

    with httpx.Client(timeout=120.0) as client:
        for idx, budget in enumerate(budgets, start=1):
            source_path = f"data/budgets_sample.json#{budget['budget_id']}"
            payload = {
                "source_path": source_path,
                "document_type": "historical_budget",
                "content": budget,
            }
            response = client.post(f"{base_url}/embeddings/ingest", json=payload)
            if response.status_code == 200:
                success += 1
                data = response.json()
                print(
                    f"[{idx:02d}] OK  id={data['document_id']} "
                    f"chunks={data['chunks_created']} ms={data['ingestion_time_ms']}"
                )
                continue
            if response.status_code == 409:
                duplicate += 1
                detail = response.json().get("detail", {})
                print(
                    f"[{idx:02d}] DUP source={source_path} document_id={detail.get('document_id')}"
                )
                continue
            response.raise_for_status()

    print(f"\nDone. inserted={success} duplicates={duplicate} total={len(budgets)}")


if __name__ == "__main__":
    main()
