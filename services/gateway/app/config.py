"""Gateway settings.

Loaded from services/gateway/.env via an absolute path so the service behaves
identically no matter which directory uvicorn was launched from.
"""
from pathlib import Path
from typing import List
from typing_extensions import Annotated

from pydantic import field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

SERVICE_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=SERVICE_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    gateway_port: int = 8080
    gateway_cors_origins: Annotated[List[str], NoDecode] = [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ]

    # ocr-service really does listen on 8000 — see its Dockerfile. The repo-root
    # .env.example says 8001 and is wrong.
    ocr_service_url: str = "http://127.0.0.1:8000"
    reasoning_service_url: str = "http://127.0.0.1:8002"
    feedback_service_url: str = "http://127.0.0.1:8003"

    connect_timeout_s: float = 5.0
    ocr_timeout_s: float = 180.0
    ocr_scheme_timeout_s: float = 120.0
    reasoning_timeout_s: float = 240.0
    feedback_timeout_s: float = 300.0

    default_ocr_mode: str = "openai_vision"
    default_use_math_ocr: bool = False

    multi_question_policy: str = "first"
    include_unanalyzed_steps: bool = True

    max_answer_images: int = 5
    max_upload_mb: int = 10

    job_ttl_seconds: int = 1800
    job_max_concurrency: int = 2
    max_jobs: int = 200

    log_level: str = "INFO"

    @field_validator("gateway_cors_origins", mode="before")
    @classmethod
    def _split_origins(cls, value):
        """Accept a comma-separated string.

        pydantic-settings would otherwise try to JSON-decode a list-typed env
        var, which makes `A,B` a confusing parse error rather than two origins.
        """
        if isinstance(value, str):
            return [origin.strip() for origin in value.split(",") if origin.strip()]
        return value

    @field_validator("multi_question_policy")
    @classmethod
    def _check_policy(cls, value: str) -> str:
        allowed = {"first", "all", "error"}
        if value not in allowed:
            raise ValueError(f"MULTI_QUESTION_POLICY must be one of {sorted(allowed)}")
        return value

    @field_validator(
        "ocr_service_url", "reasoning_service_url", "feedback_service_url"
    )
    @classmethod
    def _strip_trailing_slash(cls, value: str) -> str:
        return value.rstrip("/")

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024


settings = Settings()
