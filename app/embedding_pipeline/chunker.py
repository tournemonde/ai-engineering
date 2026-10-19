"""Structural JSON chunker: one budget component becomes one chunk."""

from __future__ import annotations

import tiktoken

from app.embedding_pipeline.schemas import Budget, BudgetComponent, Chunk

_EMBEDDING_MODEL = "text-embedding-3-small"


class JSONStructuralChunker:
    """Split historical budgets along component boundaries."""

    def __init__(self, model: str = _EMBEDDING_MODEL) -> None:
        self._encoding = tiktoken.encoding_for_model(model)

    def chunk(self, budgets: list[Budget]) -> list[Chunk]:
        chunks: list[Chunk] = []
        for budget in budgets:
            sector = budget.client_metadata.sector
            for component in budget.components:
                text = self._format_text(budget, component)
                token_count = len(self._encoding.encode(text))
                chunks.append(
                    Chunk(
                        chunk_id=f"{budget.budget_id}::{component.component_id}",
                        text=text,
                        metadata={
                            "budget_id": budget.budget_id,
                            "component_id": component.component_id,
                            "client_sector": sector,
                            "main_technology": budget.main_technology,
                            "year": budget.year,
                            "complexity": component.complexity,
                            "estimated_hours": component.estimated_hours,
                        },
                        token_count=token_count,
                    )
                )
        return chunks

    @staticmethod
    def _format_text(budget: Budget, component: BudgetComponent) -> str:
        tech = ", ".join(component.tech_stack)
        return (
            f"[Project: {budget.project_summary}]\n"
            f"[Client sector: {budget.client_metadata.sector} | "
            f"Year: {budget.year} | Main tech: {budget.main_technology}]\n"
            f"Component: {component.name}\n"
            f"Description: {component.description}\n"
            f"Tech stack: {tech}\n"
            f"Complexity: {component.complexity}\n"
            f"Estimated hours: {component.estimated_hours}"
        )
