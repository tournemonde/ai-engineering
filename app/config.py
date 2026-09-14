"""Application settings loaded from environment variables."""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration for the CAG estimation service."""

    # snippet: settings fields
    LLM_PROVIDER: Literal["openai", "anthropic"] = "openai"
    LLM_MODEL: str = ""
    OPENAI_API_KEY: str | None = None
    ANTHROPIC_API_KEY: str | None = None
    APP_ENV: str = "development"
    LOG_LEVEL: str = "DEBUG"
    MAX_OUTPUT_TOKENS: int = Field(default=4000, ge=256, le=16000)

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # snippet: provider key validation
    @model_validator(mode="after")
    def validate_provider_credentials(self) -> Settings:
        if not self.LLM_MODEL:
            if self.LLM_PROVIDER == "openai":
                self.LLM_MODEL = "gpt-4o-mini"
            else:
                self.LLM_MODEL = "claude-haiku-4-5"

        if self.LLM_PROVIDER == "openai" and not self.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is required when LLM_PROVIDER=openai")
        if self.LLM_PROVIDER == "anthropic" and not self.ANTHROPIC_API_KEY:
            raise ValueError("ANTHROPIC_API_KEY is required when LLM_PROVIDER=anthropic")
        return self


@lru_cache
def get_settings() -> Settings:
    """Return cached settings singleton."""
    return Settings()
