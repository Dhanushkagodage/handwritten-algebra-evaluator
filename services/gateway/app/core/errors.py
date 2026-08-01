"""Gateway error hierarchy and upstream error parsing.

The three module services report failures in three different shapes:

  1. reasoning-service envelope   {"error": true, "error_code", "message", "details", "status_code"}
  2. FastAPI HTTPException        {"detail": "some message"}          (ocr + feedback)
  3. FastAPI/Pydantic validation  {"detail": [{"loc", "msg", "type"}]}

`parse_upstream_error` collapses all three into one tuple so the gateway can
present a single, predictable envelope to the frontend.
"""
from typing import Any, Dict, Optional, Tuple

import httpx


class GatewayError(Exception):
    """Base class — every handled failure maps onto one ErrorEnvelope."""

    status_code: int = 500
    error_code: str = "INTERNAL_ERROR"

    def __init__(
        self,
        message: str,
        *,
        details: Optional[Dict[str, Any]] = None,
        stage: Optional[str] = None,
        error_code: Optional[str] = None,
        status_code: Optional[int] = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}
        self.stage = stage
        if error_code is not None:
            self.error_code = error_code
        if status_code is not None:
            self.status_code = status_code


class InvalidInputError(GatewayError):
    status_code = 400
    error_code = "INVALID_INPUT"


class PayloadTooLargeError(GatewayError):
    status_code = 413
    error_code = "PAYLOAD_TOO_LARGE"


class NoStepsExtractedError(GatewayError):
    status_code = 422
    error_code = "NO_STEPS_EXTRACTED"


class MarkingSchemeInvalidError(GatewayError):
    status_code = 422
    error_code = "MARKING_SCHEME_INVALID"


class MultipleQuestionsError(GatewayError):
    status_code = 422
    error_code = "MULTIPLE_QUESTIONS_DETECTED"


class JobNotFoundError(GatewayError):
    status_code = 404
    error_code = "JOB_NOT_FOUND"


class UpstreamUnavailableError(GatewayError):
    """The service process could not be reached at all."""

    status_code = 503
    error_code = "SERVICE_UNAVAILABLE"


class UpstreamTimeoutError(GatewayError):
    status_code = 504
    error_code = "UPSTREAM_TIMEOUT"


class UpstreamError(GatewayError):
    """The service answered, but with a non-2xx status."""

    status_code = 502
    error_code = "UPSTREAM_FAILED"


class UpstreamBadResponseError(GatewayError):
    """The service answered 2xx with a body we could not understand."""

    status_code = 502
    error_code = "UPSTREAM_BAD_RESPONSE"


def parse_upstream_error(response: httpx.Response) -> Tuple[Optional[str], str, Dict[str, Any]]:
    """Return (upstream_error_code, human_message, details) for any error body."""
    try:
        body = response.json()
    except ValueError:
        text = (response.text or "").strip()
        return None, text[:500] or f"HTTP {response.status_code}", {}

    if isinstance(body, dict):
        # 1. reasoning-service envelope
        if "message" in body and ("error_code" in body or body.get("error") is True):
            return (
                body.get("error_code"),
                str(body.get("message")),
                body.get("details") or {},
            )

        detail = body.get("detail")

        # 2. plain HTTPException
        if isinstance(detail, str):
            return None, detail, {}

        # 3. Pydantic validation errors
        if isinstance(detail, list):
            parts = []
            for item in detail:
                if not isinstance(item, dict):
                    parts.append(str(item))
                    continue
                loc = ".".join(str(piece) for piece in item.get("loc", []))
                parts.append(f"{loc}: {item.get('msg', '')}".strip(": "))
            message = "; ".join(part for part in parts if part)
            return "VALIDATION_ERROR", message or "Request validation failed.", {"errors": detail}

    return None, str(body)[:500], {}
