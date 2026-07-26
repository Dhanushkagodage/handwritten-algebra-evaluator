"""
Agent 1: Step Validation Agent
─────────────────────────────
INPUT : question_text + student_steps  (NO marking scheme access)
OUTPUT: StepValidationOutput — mathematical validity per step + missing transitions

Runs in parallel with Agent 2 and Agent 3 inside the LangGraph workflow.
"""
import json
import logging
import re
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import ValidationError

from app.schemas.output_schema import StepValidationOutput
from app.services.llm_factory import get_cached_llm
from app.config.settings import settings

logger = logging.getLogger(__name__)

# ── Load system prompt from disk (plain text, no f-string interpolation) ──────
_PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "step_validation.txt"
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
    """Compact JSON serialisation — minimal tokens for LLM input."""
    return json.dumps(data, separators=(",", ":"), ensure_ascii=False)


def step_validation_agent(state: dict) -> dict:
    """
    LangGraph node: validates each student step mathematically.
    Retries up to MAX_RETRIES times on invalid JSON or schema violations.
    """
    question_text: str = state["question_text"]
    student_steps: list = state["student_steps"]

    llm = get_cached_llm(
        provider=settings.llm_provider,
        model=settings.get_step_check_model(),
        temperature=settings.llm_temperature,
    )

    human_text = (
        f"Question:\n{question_text}\n\n"
        f"Student Steps:\n{_j(student_steps)}\n\n"
        "EVALUATION REMINDER: Before marking any step incorrect, mentally compute all "
        "arithmetic in that step (e.g., evaluate powers like (-2)^3 = -8, products like "
        "2×(-8) = -16, etc.) and verify mathematical equivalence of LHS and RHS after "
        "full expansion. An unevaluated but mathematically correct form is NOT an error."
    )

    last_error: Exception | None = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            logger.info("[StepValidationAgent] Attempt %d/%d", attempt, MAX_RETRIES)
            messages = [
                SystemMessage(content=_SYSTEM_PROMPT),
                HumanMessage(content=human_text),
            ]
            response = llm.invoke(messages)
            raw = _extract_json(response.content)
            validated = StepValidationOutput(**raw)
            logger.info(
                "[StepValidationAgent] Success — %d steps validated",
                len(validated.step_validations),
            )
            return {"step_validation_output": validated.model_dump()}

        except (json.JSONDecodeError, ValidationError, ValueError) as exc:
            last_error = exc
            logger.warning(
                "[StepValidationAgent] Attempt %d failed: %s", attempt, exc
            )

    # Fallback: mark all steps as unclear with low confidence
    logger.error(
        "[StepValidationAgent] All %d attempts failed. Using fallback. Last error: %s",
        MAX_RETRIES,
        last_error,
    )
    fallback = StepValidationOutput(
        step_validations=[
            {
                "step_id": s["step_id"],
                "is_valid": False,
                "status": "unclear",
                "error": f"Agent failed to evaluate: {last_error}",
                "confidence": 0.0,
            }
            for s in student_steps
        ],
        missing_transitions=["Unable to determine — agent evaluation failed"],
    )
    return {"step_validation_output": fallback.model_dump()}
