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

# A step lists multiple candidate roots if it contains two or more "x = <number>"
# assignments (e.g. "x = 5 or x = -1", "x = 5, x = -1").
_MULTI_ROOT_RE = re.compile(r"x\s*=\s*[-+±]?\s*\d")

# Domain/extraneous-root rejection language — indicates the LLM rejected the step
# for a restriction that only applies to a FINAL answer, not to a step that is
# merely listing candidate roots from the zero-product property.
_DOMAIN_REJECT_RE = re.compile(
    r"domain|not valid since|only valid|extraneous|reject|out of (the )?domain|"
    r"x\s*[<>]|undefined for|restrict",
    re.IGNORECASE,
)


def _fix_premature_domain_rejections(validated: StepValidationOutput, student_steps: list) -> None:
    """
    Deterministic guardrail (does not depend on the LLM following the prompt):
    a step that only LISTS candidate roots (e.g. "x = 5 or x = -1" from a factored
    equation) must not be marked incorrect solely because a domain restriction will
    later exclude one of them. That filtering is the job of whichever later step
    actually applies the restriction — this step is judged only on its own algebra.
    """
    content_map = {s["step_id"]: s.get("content", "") for s in student_steps}
    for result in validated.step_validations:
        if result.status != "incorrect" or not result.error:
            continue
        content = content_map.get(result.step_id, "")
        if len(_MULTI_ROOT_RE.findall(content)) >= 2 and _DOMAIN_REJECT_RE.search(result.error):
            logger.warning(
                "[StepValidationAgent] Overriding premature domain rejection on step %d "
                "(content=%r, error=%r)",
                result.step_id, content, result.error,
            )
            result.status = "correct"
            result.is_valid = True
            result.error = None


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


async def step_validation_agent(state: dict) -> dict:
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
        "full expansion. An unevaluated but mathematically correct form is NOT an error. "
        "For completing the square on ax²+bx+c=0 (when a ≠ 1), remember to divide by a first, "
        "so the term is (x + b/(2a))². Always test equivalence by expanding (x-h)² = k."
    )

    last_error: Exception | None = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            logger.info("[StepValidationAgent] Attempt %d/%d", attempt, MAX_RETRIES)
            messages = [
                SystemMessage(content=_SYSTEM_PROMPT),
                HumanMessage(content=human_text),
            ]
            response = await llm.ainvoke(messages)
            raw = _extract_json(response.content)
            validated = StepValidationOutput(**raw)
            _fix_premature_domain_rejections(validated, student_steps)
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
