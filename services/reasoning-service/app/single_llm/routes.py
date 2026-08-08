"""
routes.py (Single LLM Baseline)
───────────────────────────────
FastAPI APIRouter for Single LLM Baseline evaluation endpoint:
POST /api/v1/evaluate/single-llm         → full EvaluationOutput
POST /api/v1/evaluate/single-llm/summary → FriendlyEvaluation
"""
import json
import logging
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Body, Query

from app.schemas.input_schema import EvaluationRequest
from app.schemas.output_schema import EvaluationOutput, FriendlyEvaluation, FriendlyStep
from app.single_llm.evaluator import evaluate_single_llm
from app.single_llm.csv_recorder import record_single_llm_evaluation

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["Single LLM Baseline"])


def _load_default_request() -> EvaluationRequest:
    """Load fallback test case when no request body is provided."""
    test_file = (
        Path(__file__).resolve().parent.parent.parent
        / "tests" / "evaluation_suite" / "1_inputs" / "tc_001.json"
    )
    logger.info("[SingleLLM] No request body provided. Loading %s", test_file)
    with open(test_file, "r", encoding="utf-8") as f:
        return EvaluationRequest.model_validate(json.load(f))


def _build_friendly(output: EvaluationOutput, request: EvaluationRequest) -> FriendlyEvaluation:
    """Convert an EvaluationOutput into a FriendlyEvaluation response."""
    step_content_map = {
        s.step_id: s.content for s in request.reasoning_input.student_steps
    }
    method_detected = (
        output.steps_analysis[0].method
        if output.steps_analysis and len(output.steps_analysis) > 0
        else "undetermined"
    )
    friendly_steps = [
        FriendlyStep(
            step_number=step.step_id,
            expression=step_content_map.get(step.step_id, ""),
            validity=step.status,
            marks_awarded=step.marks_awarded,
        )
        for step in output.steps_analysis
    ]
    return FriendlyEvaluation(
        question_text=request.reasoning_input.question_text,
        detected_method=method_detected,
        assigned_marks=output.total_marks,
        total_marks=output.max_marks,
        student_steps=friendly_steps,
    )


@router.post(
    "/evaluate/single-llm",
    response_model=EvaluationOutput,
    summary="Evaluate student algebra answer using Single LLM Baseline (full output)",
    description=(
        "Runs a single LLM prompt call to evaluate a handwritten A/L algebra answer "
        "against a marking scheme. Returns per-step marks, feedback, and summary. "
        "Pass ?test_case_id=tc_001 to also append results + ground truth to single LLM "
        "evaluation CSVs in tests/evaluation_suite/3_output_results/."
    ),
)
async def evaluate_single_llm_endpoint(
    request: Optional[EvaluationRequest] = Body(None),
    test_case_id: Optional[str] = Query(
        None,
        description="e.g. tc_028 — when provided, appends results to single LLM evaluation CSVs",
    ),
) -> EvaluationOutput:
    """POST /api/v1/evaluate/single-llm"""
    if request is None:
        if test_case_id:
            tc_file = (
                Path(__file__).resolve().parent.parent.parent
                / "tests" / "evaluation_suite" / "1_inputs" / f"{test_case_id}.json"
            )
            logger.info("[SingleLLM] No request body. Loading %s", tc_file)
            with open(tc_file, "r", encoding="utf-8") as f:
                request = EvaluationRequest.model_validate(json.load(f))
        else:
            request = _load_default_request()

    output = await evaluate_single_llm(request)

    if test_case_id:
        try:
            summary = record_single_llm_evaluation(
                test_case_id=test_case_id,
                question_text=request.reasoning_input.question_text,
                max_marks=request.marking_scheme.total_marks,
                api_output=output.model_dump(),
            )
            logger.info(
                "[SingleLLM] CSV record — %s | gt_loaded=%s rows=%d",
                test_case_id,
                summary.get("gt_loaded"),
                summary.get("step_rows_written", 0),
            )
        except Exception as exc:
            logger.warning("[SingleLLM] CSV record failed: %s", exc)

    return output


@router.post(
    "/evaluate/single-llm/summary",
    response_model=FriendlyEvaluation,
    summary="Evaluate student algebra answer using Single LLM Baseline (compact summary)",
)
async def evaluate_single_llm_summary_endpoint(
    request: Optional[EvaluationRequest] = Body(None)
) -> FriendlyEvaluation:
    """POST /api/v1/evaluate/single-llm/summary"""
    if request is None:
        request = _load_default_request()

    output = await evaluate_single_llm(request)
    return _build_friendly(output, request)
