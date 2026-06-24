"""
core/handlers.py
─────────────────
Global FastAPI exception handlers.

Registered once in main.py — converts every exception type into a
consistent JSON error envelope that API consumers can rely on:

{
  "error": true,
  "error_code": "INVALID_INPUT",
  "message": "student_steps cannot be empty.",
  "details": { "field": "reasoning_input.student_steps" },
  "status_code": 400
}

Handler priority (most specific → least specific):
  1. ReasoningServiceError  (all domain exceptions)
  2. RequestValidationError (Pydantic / FastAPI 422s)
  3. HTTPException          (FastAPI explicit raises)
  4. Exception              (catch-all — never expose internals)
"""
import logging
import traceback

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.core.exceptions import ReasoningServiceError

logger = logging.getLogger(__name__)


# ── Response builder ──────────────────────────────────────────────────────────

def _error_response(
    status_code: int,
    error_code: str,
    message: str,
    details: dict | None = None,
) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={
            "error": True,
            "error_code": error_code,
            "message": message,
            "details": details or {},
            "status_code": status_code,
        },
    )


# ── Handler functions ─────────────────────────────────────────────────────────

async def reasoning_service_error_handler(
    request: Request, exc: ReasoningServiceError
) -> JSONResponse:
    """Handles all domain-specific exceptions from core/exceptions.py."""
    logger.warning(
        "[%s] %s | path=%s | details=%s",
        exc.error_code,
        exc.message,
        request.url.path,
        exc.details,
    )
    return _error_response(
        status_code=exc.status_code,
        error_code=exc.error_code,
        message=exc.message,
        details=exc.details,
    )


async def validation_error_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """Handles Pydantic request body validation errors (FastAPI 422)."""
    errors = exc.errors()
    logger.warning(
        "[VALIDATION_ERROR] path=%s | errors=%s",
        request.url.path,
        errors,
    )
    # Flatten into a readable list
    readable = [
        {
            "field": " → ".join(str(loc) for loc in err["loc"]),
            "message": err["msg"],
            "type": err["type"],
        }
        for err in errors
    ]
    return _error_response(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        error_code="VALIDATION_ERROR",
        message="Request body failed schema validation.",
        details={"errors": readable},
    )


async def http_exception_handler(
    request: Request, exc: StarletteHTTPException
) -> JSONResponse:
    """Handles explicit HTTPException raises (e.g. 404, 405)."""
    logger.warning(
        "[HTTP_%d] %s | path=%s",
        exc.status_code,
        exc.detail,
        request.url.path,
    )
    return _error_response(
        status_code=exc.status_code,
        error_code=f"HTTP_{exc.status_code}",
        message=str(exc.detail),
    )


async def unhandled_exception_handler(
    request: Request, exc: Exception
) -> JSONResponse:
    """
    Catch-all handler — prevents raw tracebacks from leaking to clients.
    Always logs the full traceback server-side.
    """
    logger.error(
        "[UNHANDLED_EXCEPTION] path=%s\n%s",
        request.url.path,
        traceback.format_exc(),
    )
    return _error_response(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        error_code="INTERNAL_ERROR",
        message="An unexpected error occurred. Please try again later.",
    )


# ── Registration helper ───────────────────────────────────────────────────────

def register_exception_handlers(app: FastAPI) -> None:
    """
    Register all exception handlers on the FastAPI app.
    Call this once in main.py after creating the app instance.

    Example:
        from app.core.handlers import register_exception_handlers
        register_exception_handlers(app)
    """
    app.add_exception_handler(ReasoningServiceError, reasoning_service_error_handler)
    app.add_exception_handler(RequestValidationError, validation_error_handler)
    app.add_exception_handler(StarletteHTTPException, http_exception_handler)
    app.add_exception_handler(Exception, unhandled_exception_handler)
    logger.info("[Handlers] Global exception handlers registered")
