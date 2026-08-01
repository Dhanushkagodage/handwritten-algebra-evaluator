"""Shared HTTP plumbing: one AsyncClient, per-stage timeouts, uniform errors."""
import logging
import time
from typing import Any, Dict, Optional, Type, TypeVar

import httpx
from pydantic import BaseModel, ValidationError

from app.config import settings
from app.core.errors import (
    UpstreamBadResponseError,
    UpstreamError,
    UpstreamTimeoutError,
    UpstreamUnavailableError,
    parse_upstream_error,
)

logger = logging.getLogger(__name__)

TModel = TypeVar("TModel", bound=BaseModel)

#: Human labels used in error messages, keyed by pipeline stage.
SERVICE_LABELS = {
    "ocr": "OCR service (Module 01)",
    "reasoning": "Reasoning service (Module 02)",
    "feedback": "Feedback service (Module 03)",
}


def build_http_client() -> httpx.AsyncClient:
    """Create the process-wide AsyncClient.

    `trust_env=False` matters on Windows: a corporate HTTP_PROXY would otherwise
    be applied to loopback calls and break every hop.
    """
    return httpx.AsyncClient(
        # Retries here cover connection establishment only — never a response.
        # That is exactly what we want while a service is still booting.
        transport=httpx.AsyncHTTPTransport(retries=2),
        limits=httpx.Limits(
            max_connections=20, max_keepalive_connections=10, keepalive_expiry=120.0
        ),
        timeout=httpx.Timeout(
            connect=settings.connect_timeout_s, read=60.0, write=30.0, pool=5.0
        ),
        follow_redirects=False,
        trust_env=False,
        headers={"user-agent": "algebra-gateway/1.0"},
    )


def stage_timeout(read_seconds: float) -> httpx.Timeout:
    return httpx.Timeout(
        connect=settings.connect_timeout_s, read=read_seconds, write=60.0, pool=5.0
    )


async def request_json(
    client: httpx.AsyncClient,
    *,
    method: str,
    url: str,
    stage: str,
    timeout: httpx.Timeout,
    json_body: Optional[Dict[str, Any]] = None,
    data: Optional[Dict[str, Any]] = None,
    files: Optional[Any] = None,
) -> Any:
    """Perform one upstream call, normalising every failure mode."""
    label = SERVICE_LABELS.get(stage, stage)
    started = time.perf_counter()

    try:
        response = await client.request(
            method, url, json=json_body, data=data, files=files, timeout=timeout
        )
    except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
        raise UpstreamUnavailableError(
            f"The {label} is not reachable at {url}. Is it running?",
            stage=stage,
            details={"url": url, "cause": str(exc)},
        ) from exc
    except httpx.TimeoutException as exc:
        raise UpstreamTimeoutError(
            f"The {label} did not respond within {timeout.read:.0f}s.",
            stage=stage,
            details={"url": url, "timeout_s": timeout.read, "cause": str(exc)},
        ) from exc
    except httpx.HTTPError as exc:
        raise UpstreamUnavailableError(
            f"The call to the {label} failed: {exc}",
            stage=stage,
            details={"url": url, "cause": str(exc)},
        ) from exc

    elapsed_ms = int((time.perf_counter() - started) * 1000)
    logger.info("[%s] %s %s -> %s in %dms", stage, method, url, response.status_code, elapsed_ms)

    if response.is_error:
        upstream_code, message, details = parse_upstream_error(response)
        raise UpstreamError(
            message,
            error_code=f"{stage.upper()}_FAILED",
            stage=stage,
            details={
                "upstream_status": response.status_code,
                "upstream_error_code": upstream_code,
                "upstream_url": url,
                **details,
            },
        )

    try:
        return response.json()
    except ValueError as exc:
        raise UpstreamBadResponseError(
            f"The {label} returned a non-JSON response.",
            error_code=f"{stage.upper()}_BAD_RESPONSE",
            stage=stage,
            details={"url": url, "body": (response.text or "")[:500]},
        ) from exc


def parse_model(payload: Any, model: Type[TModel], *, stage: str) -> TModel:
    """Validate an upstream body, converting failures into a 502."""
    label = SERVICE_LABELS.get(stage, stage)
    try:
        return model.model_validate(payload)
    except ValidationError as exc:
        raise UpstreamBadResponseError(
            f"The {label} returned a response the gateway could not understand.",
            error_code=f"{stage.upper()}_BAD_RESPONSE",
            stage=stage,
            details={"errors": exc.errors()[:10]},
        ) from exc
