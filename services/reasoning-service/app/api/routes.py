"""
FastAPI router — /evaluate endpoints
─────────────────────────────────────
POST /api/v1/evaluate         → full EvaluationOutput  (detailed, with agent internals)
POST /api/v1/evaluate/summary → FriendlyEvaluation     (compact, human-readable)

Both endpoints delegate execution to the EvaluationController.
"""
from typing import Optional
from fastapi import APIRouter
from app.schemas.input_schema import EvaluationRequest
from app.schemas.output_schema import EvaluationOutput, FriendlyEvaluation
from app.controllers import evaluation_controller

router = APIRouter(prefix="/api/v1", tags=["Evaluation"])

# Backward-compatibility alias for internal suite runners / external scripts
_run_pipeline = evaluation_controller.run_pipeline


# ── Endpoint 1: Full detailed output ──────────────────────────────────────────

@router.post(
    "/evaluate",
    response_model=EvaluationOutput,
    summary="Evaluate student algebra answer (full output)",
    description=(
        "Runs a multi-agent parallel LangGraph pipeline to evaluate a handwritten "
        "algebra answer against a marking scheme. Returns the full per-step "
        "marks, agent internals, method feedback, and a summary."
    ),
)
async def evaluate(request: Optional[EvaluationRequest] = None) -> EvaluationOutput:
    """POST /api/v1/evaluate — returns full EvaluationOutput via EvaluationController."""
    return await evaluation_controller.evaluate(request)


# ── Endpoint 2: Compact friendly output ───────────────────────────────────────

@router.post(
    "/evaluate/summary",
    response_model=FriendlyEvaluation,
    summary="Evaluate student algebra answer (compact summary)",
    description=(
        "Runs the same LangGraph pipeline as /evaluate but returns a compact, "
        "human-readable summary: question, method, total marks, and per-step "
        "validity + marks only. No internal agent details."
    ),
)
async def evaluate_summary(request: Optional[EvaluationRequest] = None) -> FriendlyEvaluation:
    """POST /api/v1/evaluate/summary — returns compact FriendlyEvaluation via EvaluationController."""
    return await evaluation_controller.evaluate_summary(request)
