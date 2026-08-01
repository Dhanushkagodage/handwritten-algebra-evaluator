"""Client for reasoning-service (Module 02).

We call `/api/v1/evaluate` rather than `/api/v1/evaluate/summary`: the full
output carries `step_validation` (per-step error descriptions) and
`method_detection`, both of which the feedback stage needs. The summary
endpoint also passes `status` straight through, so `partially_correct` and
`unclear` would 422 against feedback-service's three-value enum.
"""
from typing import Any, Dict

import httpx

from app.clients.base import parse_model, request_json, stage_timeout
from app.config import settings
from app.schemas.upstream_reasoning import EvaluationOutput, EvaluationRequest

STAGE = "reasoning"


async def evaluate(
    client: httpx.AsyncClient, *, request: EvaluationRequest
) -> EvaluationOutput:
    payload = await request_json(
        client,
        method="POST",
        url=f"{settings.reasoning_service_url}/api/v1/evaluate",
        stage=STAGE,
        timeout=stage_timeout(settings.reasoning_timeout_s),
        json_body=request.model_dump(mode="json"),
    )
    return parse_model(payload, EvaluationOutput, stage=STAGE)


async def ping(client: httpx.AsyncClient) -> Dict[str, Any]:
    response = await client.get(
        f"{settings.reasoning_service_url}/health", timeout=httpx.Timeout(3.0)
    )
    response.raise_for_status()
    body = response.json()
    return body if isinstance(body, dict) else {}
