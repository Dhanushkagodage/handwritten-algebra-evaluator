"""One error envelope for every failure path."""
import logging

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.core.errors import GatewayError
from app.schemas.gateway import ErrorEnvelope

logger = logging.getLogger(__name__)


def envelope_from_gateway_error(error: GatewayError) -> ErrorEnvelope:
    return ErrorEnvelope(
        error_code=error.error_code,
        message=error.message,
        detail=error.message,
        stage=error.stage,
        status_code=error.status_code,
        details=error.details,
    )


def _json(envelope: ErrorEnvelope) -> JSONResponse:
    return JSONResponse(status_code=envelope.status_code, content=envelope.model_dump(mode="json"))


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(GatewayError)
    async def _handle_gateway_error(_: Request, exc: GatewayError):
        logger.warning("[%s] %s — %s", exc.error_code, exc.stage or "gateway", exc.message)
        return _json(envelope_from_gateway_error(exc))

    @app.exception_handler(RequestValidationError)
    async def _handle_validation_error(_: Request, exc: RequestValidationError):
        return _json(
            ErrorEnvelope(
                error_code="VALIDATION_ERROR",
                message="The request was rejected by the gateway.",
                detail="The request was rejected by the gateway.",
                status_code=422,
                details={"errors": exc.errors()},
            )
        )

    @app.exception_handler(StarletteHTTPException)
    async def _handle_http_exception(_: Request, exc: StarletteHTTPException):
        message = str(exc.detail)
        return _json(
            ErrorEnvelope(
                error_code="HTTP_ERROR",
                message=message,
                detail=message,
                status_code=exc.status_code,
            )
        )

    @app.exception_handler(Exception)
    async def _handle_unexpected(_: Request, exc: Exception):
        # Log the traceback, never leak it to the caller.
        logger.exception("Unhandled gateway error: %s", exc)
        message = "The gateway hit an unexpected error. Check the gateway logs."
        return _json(
            ErrorEnvelope(
                error_code="INTERNAL_ERROR",
                message=message,
                detail=message,
                status_code=500,
            )
        )
