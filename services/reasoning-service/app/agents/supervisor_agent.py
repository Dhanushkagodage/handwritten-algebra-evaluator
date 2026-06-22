"""
Supervisor Agent
────────────────
INPUT : step_validation_output + method_detection_output + scheme_matching_output + marking_scheme
OUTPUT: EvaluationOutput — per-step marks, totals, and narrative feedback

Runs AFTER all three parallel agents complete in the LangGraph workflow.
"""
import json
import logging
import re
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import ValidationError

from app.schemas.output_schema import EvaluationOutput
from app.services.llm_factory import get_cached_llm

logger = logging.getLogger(__name__)

# ── Load system prompt from disk (plain text, no f-string interpolation) ──────
_PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "supervisor.txt"
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


def _build_fallback_output(
    marking_scheme: dict,
    step_validation: dict,
    scheme_matching: dict,
    method_detection: dict,
) -> EvaluationOutput:
    """
    Deterministic fallback when the LLM supervisor fails.
    Uses scheme match scores directly to compute marks without LLM judgment.
    """
    logger.warning("[SupervisorAgent] Using deterministic fallback mark computation")

    validation_map = {
        v["step_id"]: v
        for v in step_validation.get("step_validations", [])
    }
    match_map = {
        m["step_id"]: m
        for m in scheme_matching.get("step_matches", [])
    }
    scheme_mark_map = {
        s["step_no"]: s["marks"]
        for s in marking_scheme["steps"]
    }

    steps_analysis = []
    total_marks = 0.0

    for match in scheme_matching.get("step_matches", []):
        step_id = match["step_id"]
        val = validation_map.get(step_id, {})
        score = match.get("match_score", 0.0)
        matched_no = match.get("matched_scheme_step")
        max_m = scheme_mark_map.get(matched_no, 0.0) if matched_no else 0.0

        raw = score * max_m
        awarded = round(raw * 2) / 2
        awarded = max(0.0, min(awarded, max_m))
        total_marks += awarded

        steps_analysis.append({
            "step_id": step_id,
            "validity": val.get("is_valid", False),
            "status": val.get("status", "unclear"),
            "method": method_detection.get("detected_method", "undetermined"),
            "matched_scheme_step": matched_no,
            "match_score": score,
            "marks_awarded": awarded,
            "max_marks": max_m,
            "reason": "Computed via deterministic fallback (LLM supervisor failed)",
            "confidence": 0.3,
        })

    max_marks = marking_scheme.get("total_marks", 0.0)
    percentage = round((total_marks / max_marks) * 100, 1) if max_marks > 0 else 0.0

    return EvaluationOutput(
        steps_analysis=steps_analysis,
        total_marks=total_marks,
        max_marks=max_marks,
        percentage=percentage,
        summary="Evaluation completed using fallback logic (supervisor LLM unavailable).",
        method_feedback=f"Detected method: {method_detection.get('detected_method', 'undetermined')}.",
        missing_steps_feedback=None,
    )


def supervisor_agent(state: dict) -> dict:
    """
    LangGraph node: synthesizes all agent outputs into a final evaluation.
    Falls back to deterministic mark computation if LLM fails MAX_RETRIES times.
    """
    step_validation: dict = state["step_validation_output"]
    method_detection: dict = state["method_detection_output"]
    scheme_matching: dict = state["scheme_matching_output"]
    marking_scheme: dict = state["marking_scheme"]

    llm = get_cached_llm()

    human_text = (
        f"## Step Validation Agent Output\n{json.dumps(step_validation, indent=2)}\n\n"
        f"## Method Detection Agent Output\n{json.dumps(method_detection, indent=2)}\n\n"
        f"## Scheme Matching Agent Output\n{json.dumps(scheme_matching, indent=2)}\n\n"
        f"## Official Marking Scheme\n{json.dumps(marking_scheme, indent=2)}"
    )

    last_error: Exception | None = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            logger.info("[SupervisorAgent] Attempt %d/%d", attempt, MAX_RETRIES)
            messages = [
                SystemMessage(content=_SYSTEM_PROMPT),
                HumanMessage(content=human_text),
            ]
            response = llm.invoke(messages)
            raw = _extract_json(response.content)
            validated = EvaluationOutput(**raw)
            logger.info(
                "[SupervisorAgent] Success — total_marks=%.1f/%.1f (%.1f%%)",
                validated.total_marks,
                validated.max_marks,
                validated.percentage,
            )
            return {"evaluation_output": validated.model_dump()}

        except (json.JSONDecodeError, ValidationError, ValueError) as exc:
            last_error = exc
            logger.warning(
                "[SupervisorAgent] Attempt %d failed: %s", attempt, exc
            )

    # Deterministic fallback — never return empty-handed
    logger.error(
        "[SupervisorAgent] All %d attempts failed. Using deterministic fallback. Last error: %s",
        MAX_RETRIES,
        last_error,
    )
    fallback = _build_fallback_output(
        marking_scheme, step_validation, scheme_matching, method_detection
    )
    return {"evaluation_output": fallback.model_dump()}
