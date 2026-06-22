"""
core/exceptions.py
───────────────────
Domain-specific exception hierarchy for the reasoning service.

All custom exceptions carry:
  - message     : human-readable description
  - status_code : the HTTP status code to return
  - error_code  : machine-readable slug (for API consumers / logging)
  - details     : optional extra context dict

Raise these from agents / services — the global handlers in core/handlers.py
will convert them automatically to structured JSON responses.
"""
from typing import Any


class ReasoningServiceError(Exception):
    """Base exception for all reasoning-service errors."""

    status_code: int = 500
    error_code: str = "INTERNAL_ERROR"

    def __init__(
        self,
        message: str = "An unexpected error occurred.",
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}


# ── 400 Bad Request ───────────────────────────────────────────────────────────

class InvalidInputError(ReasoningServiceError):
    """Raised when the request payload fails domain validation."""
    status_code = 400
    error_code = "INVALID_INPUT"


class EmptyStepsError(InvalidInputError):
    """Raised when student_steps list is empty."""
    error_code = "EMPTY_STEPS"

    def __init__(self) -> None:
        super().__init__(
            message="student_steps cannot be empty.",
            details={"field": "reasoning_input.student_steps"},
        )


class EmptySchemeError(InvalidInputError):
    """Raised when marking scheme has no steps."""
    error_code = "EMPTY_SCHEME"

    def __init__(self) -> None:
        super().__init__(
            message="marking_scheme.steps cannot be empty.",
            details={"field": "marking_scheme.steps"},
        )


# ── 422 Unprocessable Entity ──────────────────────────────────────────────────

class SchemaParseError(ReasoningServiceError):
    """Raised when an LLM response cannot be parsed into the expected schema."""
    status_code = 422
    error_code = "SCHEMA_PARSE_ERROR"

    def __init__(self, agent: str, raw_response: str, cause: str) -> None:
        super().__init__(
            message=f"Agent '{agent}' returned an unparseable response after all retries.",
            details={
                "agent": agent,
                "raw_response_snippet": raw_response[:300],
                "cause": cause,
            },
        )


# ── 503 Service Unavailable ───────────────────────────────────────────────────

class LLMUnavailableError(ReasoningServiceError):
    """Raised when the LLM provider is unreachable or returns an auth error."""
    status_code = 503
    error_code = "LLM_UNAVAILABLE"

    def __init__(self, provider: str, cause: str) -> None:
        super().__init__(
            message=f"LLM provider '{provider}' is unavailable.",
            details={"provider": provider, "cause": cause},
        )


class LLMConfigError(ReasoningServiceError):
    """Raised when LLM configuration (API key, model name) is invalid."""
    status_code = 503
    error_code = "LLM_CONFIG_ERROR"


# ── 500 Internal Server Error ─────────────────────────────────────────────────

class PipelineError(ReasoningServiceError):
    """Raised when the LangGraph pipeline fails in an unrecoverable way."""
    status_code = 500
    error_code = "PIPELINE_ERROR"

    def __init__(self, cause: str) -> None:
        super().__init__(
            message="The evaluation pipeline encountered an unrecoverable error.",
            details={"cause": cause},
        )


class AgentError(ReasoningServiceError):
    """Raised when a specific agent fails all retries and has no fallback."""
    status_code = 500
    error_code = "AGENT_ERROR"

    def __init__(self, agent: str, cause: str) -> None:
        super().__init__(
            message=f"Agent '{agent}' failed after all retries.",
            details={"agent": agent, "cause": cause},
        )
