"""
FastAPI router — /evaluate endpoints
─────────────────────────────────────
POST /api/v1/evaluate         → full EvaluationOutput  (detailed, with agent internals)
POST /api/v1/evaluate/summary → FriendlyEvaluation     (compact, human-readable)

Both endpoints run the same LangGraph pipeline.
The only difference is the response format returned to the caller.
"""
import logging
import json
from typing import Optional, Tuple
from fastapi import APIRouter, Body, Query
from pathlib import Path
from app.core.exceptions import EmptySchemeError, EmptyStepsError, PipelineError
from app.schemas.input_schema import EvaluationRequest
from app.schemas.output_schema import EvaluationOutput, FriendlyEvaluation, FriendlyStep
from app.services.langgraph_flow import build_graph
from app.services.csv_recorder import record_evaluation

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["Evaluation"])

# Compile graph once at import time (not per-request)
_graph = build_graph()


# ── Shared helpers ─────────────────────────────────────────────────────────────

def _load_default_request() -> EvaluationRequest:
    """Load fallback test case when no request body is provided."""
    test_file = (
        Path(__file__).resolve().parent.parent.parent
        / "tests" / "evaluation_suite" / "1_inputs" / "tc_001.json"
    )
    logger.info("[/evaluate] No request body provided. Loading %s", test_file)
    with open(test_file, "r", encoding="utf-8") as f:
        return EvaluationRequest.model_validate(json.load(f))


async def _run_pipeline(request: EvaluationRequest) -> Tuple[dict, dict, dict]:
    """
    Validate the request, invoke the LangGraph pipeline, and return
    (output_dict, state_dict, result_dict) for the caller to format as needed.
    """
    if not request.reasoning_input.student_steps:
        raise EmptyStepsError()
    if not request.marking_scheme.steps:
        raise EmptySchemeError()

    logger.info(
        "[pipeline] Received request — steps=%d, scheme_steps=%d",
        len(request.reasoning_input.student_steps),
        len(request.marking_scheme.steps),
    )

    state = {
        "question_text": request.reasoning_input.question_text,
        "student_steps": [s.model_dump() for s in request.reasoning_input.student_steps],
        "final_answer": request.reasoning_input.final_answer,
        "marking_scheme": request.marking_scheme.model_dump(),
    }

    try:
        result = await _graph.ainvoke(state)
    except Exception as exc:
        logger.exception("[pipeline] Graph invocation failed: %s", exc)
        raise PipelineError(cause=str(exc))

    output = result.get("evaluation_output")
    if output is None:
        logger.error("[pipeline] evaluation_output missing from graph result")
        raise PipelineError(cause="No evaluation_output key in graph result")

    # Inject intermediate agent outputs for callers that need them
    output["step_validation"] = result.get("step_validation_output")
    output["method_detection"] = result.get("method_detection_output")
    output["scheme_matching"] = result.get("scheme_matching_output")

    logger.info(
        "[pipeline] Evaluation complete — %.1f/%.1f marks (%.1f%%)",
        output["total_marks"],
        output["max_marks"],
        output["percentage"],
    )

    return output, state, result


def _build_friendly(output: dict, state: dict, result: dict) -> FriendlyEvaluation:
    """Convert a pipeline output dict into a FriendlyEvaluation response."""
    step_content_map: dict = {
        s["step_id"]: s.get("content", "") for s in state["student_steps"]
    }
    method_detected: str = (
        result.get("method_detection_output", {}).get("detected_method", "undetermined")
    )
    friendly_steps = [
        FriendlyStep(
            step_number=step["step_id"],
            expression=step_content_map.get(step["step_id"], ""),
            validity=step["status"],
            marks_awarded=step["marks_awarded"],
        )
        for step in output["steps_analysis"]
    ]
    return FriendlyEvaluation(
        question_text=state["question_text"],
        detected_method=method_detected,
        assigned_marks=output["total_marks"],
        total_marks=output["max_marks"],
        student_steps=friendly_steps,
    )


# ── Endpoint 1: Full detailed output ──────────────────────────────────────────

@router.post(
    "/evaluate",
    response_model=EvaluationOutput,
    summary="Evaluate student algebra answer (full output)",
    description=(
        "Runs a 3-agent parallel LangGraph pipeline to evaluate a handwritten "
        "A/L algebra answer against a marking scheme. Returns the full per-step "
        "marks, agent internals, method feedback, and a summary. "
        "Pass ?test_case_id=tc_001 to also append results + ground truth to the "
        "evaluation CSVs in tests/evaluation_suite/3_output_results/."
    ),
)
async def evaluate(
    request: Optional[EvaluationRequest] = Body(None),
    test_case_id: Optional[str] = Query(
        None,
        description="e.g. tc_001 — when provided, appends results to evaluation CSVs",
    ),
) -> EvaluationOutput:
    """POST /api/v1/evaluate — returns full EvaluationOutput.

    Optional query param ?test_case_id=tc_NNN: after a successful evaluation
    the endpoint automatically appends one question-level row and N step-level
    rows to the CSV files in tests/evaluation_suite/3_output_results/.
    """
    if request is None:
        if test_case_id:
            # Load the specific test case input file matching the query param
            tc_file = (
                Path(__file__).resolve().parent.parent.parent
                / "tests" / "evaluation_suite" / "1_inputs" / f"{test_case_id}.json"
            )
            logger.info("[/evaluate] No request body. Loading %s", tc_file)
            with open(tc_file, "r", encoding="utf-8") as f:
                request = EvaluationRequest.model_validate(json.load(f))
        else:
            request = _load_default_request()

    output, state, result = await _run_pipeline(request)

    # ── CSV recording (only when test_case_id is supplied) ────────────────────
    if test_case_id:
        try:
            summary = record_evaluation(
                test_case_id=test_case_id,
                question_text=state["question_text"],
                max_marks=request.marking_scheme.total_marks,
                api_output=output,
            )
            logger.info(
                "[/evaluate] CSV record — %s | gt_loaded=%s rows=%d",
                test_case_id,
                summary.get("gt_loaded"),
                summary.get("step_rows_written", 0),
            )
        except Exception as exc:
            # Never let CSV writing break the evaluation response
            logger.warning("[/evaluate] CSV record failed: %s", exc)

    return EvaluationOutput(**output)


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
async def evaluate_summary(request: Optional[EvaluationRequest] = Body(None)) -> FriendlyEvaluation:
    """POST /api/v1/evaluate/summary — returns compact FriendlyEvaluation only."""
    if request is None:
        request = _load_default_request()

    output, state, result = await _run_pipeline(request)
    return _build_friendly(output, state, result)
