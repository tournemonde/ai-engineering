"""Async SQLAlchemy session helpers for Session 8 vector store."""

from __future__ import annotations

from collections.abc import AsyncIterator
from functools import lru_cache

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.config import get_settings


def _as_asyncpg_url(url: str) -> str:
    if "+asyncpg" in url:
        return url
    if "+psycopg" in url:
        return url.replace("+psycopg", "+asyncpg")
    return url.replace("postgresql://", "postgresql+asyncpg://")


@lru_cache
def get_async_engine() -> AsyncEngine:
    return create_async_engine(
        _as_asyncpg_url(get_settings().DATABASE_URL),
        pool_pre_ping=True,
    )


AsyncSessionLocal = async_sessionmaker(
    bind=get_async_engine(),
    autoflush=False,
    autocommit=False,
    expire_on_commit=False,
    class_=AsyncSession,
)


async def get_async_session() -> AsyncIterator[AsyncSession]:
    session = AsyncSessionLocal()
    try:
        yield session
    finally:
        await session.close()
