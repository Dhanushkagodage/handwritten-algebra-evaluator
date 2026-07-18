"""
FastAPI router — /evaluate endpoint
────────────────────────────────────
Accepts Input A (reasoning) + Input B (marking scheme) in a single request.
Invokes the LangGraph multi-agent pipeline and returns the structured evaluation.
"""
import logging
import json
from typing import Optional
from fastapi import APIRouter
from pathlib import Path
from app.core.exceptions import EmptySchemeError, EmptyStepsError, PipelineError
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
async def evaluate(request: Optional[EvaluationRequest] = None) -> EvaluationOutput:
    """
    POST /api/v1/evaluate

    Input A (reasoning_input): question + student steps
    Input B (marking_scheme): official marking scheme
    """
    #  the req body is not available then execute with the selected json 
    if request is None:
        # Resolve path relative to routes.py's directory for robustness
        test_file = Path(__file__).resolve().parent.parent.parent / "tests" / "test_cases" / "tc05_log_equation_correct_method.json"

        logger.info(
            "[/evaluate] No request body provided. Loading %s",
            test_file,
        )

        with open(test_file, "r", encoding="utf-8") as f:
            request = EvaluationRequest.model_validate(
                json.load(f)
            )



    # ── Domain input guards ────────────────────────────────────────────────────
    if not request.reasoning_input.student_steps:
        raise EmptyStepsError()

    if not request.marking_scheme.steps:
        raise EmptySchemeError()

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
        raise PipelineError(cause=str(exc))

    output = result.get("evaluation_output")
    if output is None:
        logger.error("[/evaluate] evaluation_output missing from graph result")
        raise PipelineError(cause="No evaluation_output key in graph result")

    # Inject intermediate agent outputs into the output dictionary for debugging/inspection
    output["step_validation"] = result.get("step_validation_output")
    output["method_detection"] = result.get("method_detection_output")
    output["scheme_matching"] = result.get("scheme_matching_output")

    logger.info(
        "[/evaluate] Evaluation complete — %.1f/%.1f marks (%.1f%%)",
        output["total_marks"],
        output["max_marks"],
        output["percentage"],
    )

    return EvaluationOutput(**output)

