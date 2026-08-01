"""Client for feedback-service (Module 03).

Never retry this call at the gateway. feedback-service already retries the
Hugging Face Space internally with a (0, 5, 15, 30)s backoff ladder; a gateway
retry would multiply a 90-second cold start and burn ZeroGPU quota. That ladder
is also why FEEDBACK_TIMEOUT_S defaults to 300 — set it lower and the gateway
504s while the downstream request is still succeeding.
"""
from typing import Any, Dict

import httpx

from app.clients.base import parse_model, request_json, stage_timeout
from app.config import settings
from app.schemas.upstream_feedback import FeedbackRequest, FeedbackResponse

STAGE = "feedback"


async def generate_feedback(
    client: httpx.AsyncClient, *, request: FeedbackRequest
) -> FeedbackResponse:
    payload = await request_json(
        client,
        method="POST",
        url=f"{settings.feedback_service_url}/api/v1/feedback",
        stage=STAGE,
        timeout=stage_timeout(settings.feedback_timeout_s),
        json_body=request.model_dump(mode="json"),
    )
    return parse_model(payload, FeedbackResponse, stage=STAGE)


async def ping(client: httpx.AsyncClient) -> Dict[str, Any]:
    response = await client.get(
        f"{settings.feedback_service_url}/health", timeout=httpx.Timeout(3.0)
    )
    response.raise_for_status()
    body = response.json()
    return body if isinstance(body, dict) else {}
