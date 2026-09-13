"""FastAPI application entrypoint for the CAG estimator."""

from __future__ import annotations

from fastapi import FastAPI

from app.routers import estimations

app = FastAPI(
    title="Estimador CAG",
    description=(
        "Sistema de estimación de software con arquitectura CAG "
        "(contexto estático inyectado en el prompt)."
    ),
    version="0.1.0",
)

app.include_router(estimations.router)


@app.get("/health")
async def health() -> dict[str, str]:
    """Liveness probe for local runs and CI smoke checks."""
    return {"status": "healthy"}
