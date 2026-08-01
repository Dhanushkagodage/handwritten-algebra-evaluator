"""Health endpoints — the gateway's own, and a fan-out over the three modules."""
import asyncio
import time
from typing import Any, Awaitable, Callable, Dict, Tuple

import httpx
from fastapi import APIRouter, Query, Request
from fastapi.responses import JSONResponse

from app.clients import feedback as feedback_client
from app.clients import ocr as ocr_client
from app.clients import reasoning as reasoning_client
from app.config import settings
from app.schemas.gateway import ServiceHealth, ServicesHealthResponse

router = APIRouter(tags=["Health"])

FEEDBACK_HEALTH_NOTE = (
    "This only proves the feedback service process is alive. It does not check that "
    "its Hugging Face Space is reachable or awake."
)


@router.get("/health", summary="Gateway liveness (instant, no fan-out)")
async def health() -> Dict[str, str]:
    return {"status": "ok", "service": "gateway", "version": "1.0.0"}


async def _probe(
    name: str,
    url: str,
    ping: Callable[[httpx.AsyncClient], Awaitable[Dict[str, Any]]],
    client: httpx.AsyncClient,
) -> Tuple[str, ServiceHealth]:
    started = time.perf_counter()
    try:
        body = await ping(client)
    except Exception as exc:  # noqa: BLE001 — a probe reports, it never raises
        return name, ServiceHealth(status="down", url=url, error=f"{type(exc).__name__}: {exc}")

    return name, ServiceHealth(
        status="up",
        url=url,
        latency_ms=int((time.perf_counter() - started) * 1000),
        details=body,
        note=FEEDBACK_HEALTH_NOTE if name == "feedback" else None,
    )


@router.get(
    "/health/services",
    response_model=ServicesHealthResponse,
    summary="Check all three module services",
    description=(
        "Echoes the URL used for each service, so a port mismatch (the repo README "
        "documents the OCR service on 8001 while it actually listens on 8000) is "
        "self-diagnosing. Answers 200 by default; pass ?strict=1 for 503 when any "
        "service is down."
    ),
)
async def health_services(request: Request, strict: bool = Query(False)):
    client: httpx.AsyncClient = request.app.state.http

    probes = await asyncio.gather(
        # ocr-service has no /health — its health route is GET /.
        _probe("ocr", settings.ocr_service_url, ocr_client.ping, client),
        _probe("reasoning", settings.reasoning_service_url, reasoning_client.ping, client),
        _probe("feedback", settings.feedback_service_url, feedback_client.ping, client),
    )

    services = dict(probes)
    overall = "ok" if all(item.status == "up" for item in services.values()) else "degraded"
    body = ServicesHealthResponse(status=overall, services=services)

    if strict and overall != "ok":
        return JSONResponse(status_code=503, content=body.model_dump(mode="json"))
    return body
