"""
config/settings.py
──────────────────
Centralised, validated application settings using Pydantic BaseSettings.
All values are read from environment variables / .env file.

Usage anywhere in the app:
    from app.config.settings import settings
    print(settings.llm_model)
"""
from functools import lru_cache
from typing import Literal

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Service ───────────────────────────────────────────────────────────────
    app_name: str = "Handwritten Algebra Evaluator — Reasoning Service"
    app_version: str = "2.0.0"
    port: int = Field(default=8002, ge=1, le=65535)
    debug: bool = False

    # ── LLM ───────────────────────────────────────────────────────────────────
    llm_provider: Literal["openai", "groq"] = "openai"
    llm_model: str = "gpt-4o"
    llm_temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    llm_max_retries: int = Field(default=3, ge=1, le=10)

    # ── Per-Agent Models (override llm_model for individual agents) ───────────
    step_check_model: str = ""      # STEP_CHECK_MODEL  — Step Validation Agent
    method_detect_model: str = ""   # METHOD_DETECT_MODEL — Method Detection Agent
    scheme_model: str = ""          # SCHEME_MODEL       — Scheme Matching Agent
    supervisor_model: str = ""      # SUPERVISOR_MODEL   — Supervisor Agent

    def get_step_check_model(self) -> str:
        return self.step_check_model or self.llm_model

    def get_method_detect_model(self) -> str:
        return self.method_detect_model or self.llm_model

    def get_scheme_model(self) -> str:
        return self.scheme_model or self.llm_model

    def get_supervisor_model(self) -> str:
        return self.supervisor_model or self.llm_model

    # ── API Keys ──────────────────────────────────────────────────────────────
    openai_api_key: str = ""
    groq_api_key: str = ""

    # ── CORS ──────────────────────────────────────────────────────────────────
    cors_origins: list[str] = ["*"]

    @model_validator(mode="after")
    def _validate_api_key(self) -> "Settings":
        """Ensure the correct API key is present for the chosen provider."""
        if self.llm_provider == "openai" and not self.openai_api_key:
            raise ValueError(
                "OPENAI_API_KEY must be set when LLM_PROVIDER=openai"
            )
        if self.llm_provider == "groq" and not self.groq_api_key:
            raise ValueError(
                "GROQ_API_KEY must be set when LLM_PROVIDER=groq"
            )
        return self


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the singleton Settings instance (cached after first call)."""
    return Settings()


# Module-level singleton — import this everywhere
settings: Settings = get_settings()
