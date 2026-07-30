"""
evaluator.py (Single LLM Baseline)
───────────────────────────────────
Single LLM evaluation service for stepwise algebra grading.
Executes step validity, method identification, scheme matching,
and mark allocation in a single LLM invocation.
"""
import json
import logging
import re
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import ValidationError

from app.config.settings import settings
from app.schemas.input_schema import EvaluationRequest
from app.schemas.output_schema import EvaluationOutput, StepAnalysis
from app.services.llm_factory import get_cached_llm

logger = logging.getLogger(__name__)

# Load system prompt
_PROMPT_PATH = Path(__file__).parent / "prompt.txt"
_SYSTEM_PROMPT: str = _PROMPT_PATH.read_text(encoding="utf-8")

MAX_RETRIES = 3


def _extract_json(text: str) -> dict:
    """Extract first JSON object from text, handling markdown fences."""
    clean = re.sub(r"```(?:json)?\s*", "", text).replace("```", "").strip()
    start = clean.find("{")
    if start == -1:
        raise ValueError("No JSON object found in LLM response")
    depth, end = 0, start
    for i, ch in enumerate(clean[start:], start=start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end = i
                break
    return json.loads(clean[start : end + 1])


def _j(data) -> str:
    """Compact JSON serialisation."""
    return json.dumps(data, separators=(",", ":"), ensure_ascii=False)


def _build_fallback_output(request: EvaluationRequest) -> EvaluationOutput:
    """Deterministic fallback output when single LLM call fails completely."""
    logger.warning("[SingleLLMEvaluator] Using deterministic fallback output")
    student_steps = request.reasoning_input.student_steps
    max_marks = float(request.marking_scheme.total_marks)
    
    steps_analysis = [
        StepAnalysis(
            step_id=step.step_id,
            validity=False,
            status="unclear",
            method="undetermined",
            matched_scheme_step=None,
            match_score=0.0,
            marks_awarded=0.0,
            max_marks=0.0,
            reason="Fallback — single LLM evaluation failed.",
            confidence=0.0,
        )
        for step in student_steps
    ]
    
    return EvaluationOutput(
        steps_analysis=steps_analysis,
        total_marks=0.0,
        max_marks=max_marks,
        percentage=0.0,
        summary="Evaluation failed to execute via single LLM.",
        method_feedback="Single LLM evaluation error.",
        missing_steps_feedback=None,
    )


async def evaluate_single_llm(request: EvaluationRequest) -> EvaluationOutput:
    """
    Evaluates a student algebra solution using a Single LLM call.
    """
    question_text = request.reasoning_input.question_text
    student_steps = [s.model_dump() for s in request.reasoning_input.student_steps]
    final_answer = request.reasoning_input.final_answer
    marking_scheme = request.marking_scheme.model_dump()

    llm = get_cached_llm(
        provider=settings.llm_provider,
        model=settings.get_supervisor_model(),
        temperature=settings.llm_temperature,
    )

    human_text = (
        f"Question Text:\n{question_text}\n\n"
        f"Student Steps:\n{_j(student_steps)}\n\n"
        f"Final Answer:\n{final_answer}\n\n"
        f"Official Marking Scheme:\n{_j(marking_scheme)}"
    )

    last_error: Exception | None = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            logger.info("[SingleLLMEvaluator] Attempt %d/%d", attempt, MAX_RETRIES)
            messages = [
                SystemMessage(content=_SYSTEM_PROMPT),
                HumanMessage(content=human_text),
            ]
            response = await llm.ainvoke(messages)
            raw = _extract_json(response.content)
            
            # Ensure total_marks and percentage are consistent
            output = EvaluationOutput(**raw)
            output.total_marks = round(sum(s.marks_awarded for s in output.steps_analysis), 2)
            output.total_marks = min(output.total_marks, output.max_marks)
            output.percentage = (
                round((output.total_marks / output.max_marks) * 100, 1)
                if output.max_marks > 0 else 0.0
            )

            logger.info(
                "[SingleLLMEvaluator] Success — total_marks=%.1f/%.1f (%.1f%%)",
                output.total_marks,
                output.max_marks,
                output.percentage,
            )
            return output

        except (json.JSONDecodeError, ValidationError, ValueError) as exc:
            last_error = exc
            logger.warning("[SingleLLMEvaluator] Attempt %d failed: %s", attempt, exc)

    logger.error("[SingleLLMEvaluator] All attempts failed. Returning fallback. Error: %s", last_error)
    return _build_fallback_output(request)
