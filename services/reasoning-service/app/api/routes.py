"""
FastAPI router — /evaluate endpoint
────────────────────────────────────
Accepts Input A (reasoning) + Input B (marking scheme) in a single request.
Invokes the LangGraph multi-agent pipeline and returns the structured evaluation.
"""
import logging

from fastapi import APIRouter, HTTPException

from app.schemas.input_schema import EvaluationRequest
from app.schemas.output_schema import EvaluationOutput
from app.services.langgraph_flow import build_graph

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["Evaluation"])

# Compile graph once at import time (not per-request)
_graph = build_graph()


@router.post(
    "/evaluate",
    response_model=EvaluationOutput,
    summary="Evaluate student algebra answer",
    description=(
        "Runs a 3-agent parallel LangGraph pipeline to evaluate a handwritten "
        "A/L algebra answer against a marking scheme. Returns per-step marks, "
        "method feedback, and a summary."
    ),
)
async def evaluate(request: EvaluationRequest) -> EvaluationOutput:
    """
    POST /api/v1/evaluate

    Input A (reasoning_input): question + student steps
    Input B (marking_scheme): official marking scheme
    """
    logger.info(
        "[/evaluate] Received request — steps=%d, scheme_steps=%d",
        len(request.reasoning_input.student_steps),
        len(request.marking_scheme.steps),
    )

    # Build LangGraph initial state
    state = {
        "question_text": request.reasoning_input.question_text,
        "student_steps": [s.model_dump() for s in request.reasoning_input.student_steps],
        "final_answer": request.reasoning_input.final_answer,
        "marking_scheme": request.marking_scheme.model_dump(),
    }

    try:
        result = _graph.invoke(state)
    except Exception as exc:
        logger.exception("[/evaluate] Graph invocation failed: %s", exc)
        raise HTTPException(
            status_code=500,
            detail=f"Evaluation pipeline failed: {str(exc)}",
        )

    output = result.get("evaluation_output")
    if output is None:
        logger.error("[/evaluate] evaluation_output missing from graph result")
        raise HTTPException(
            status_code=500,
            detail="No evaluation output produced by the pipeline.",
        )

    logger.info(
        "[/evaluate] Evaluation complete — %.1f/%.1f marks (%.1f%%)",
        output["total_marks"],
        output["max_marks"],
        output["percentage"],
    )

    return EvaluationOutput(**output)
