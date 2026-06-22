"""
Agent 2: Method Detection Agent
────────────────────────────────
INPUT : question_text + student_steps  (NO marking scheme access)
OUTPUT: MethodDetectionOutput — detected solving method + alternatives

Runs in parallel with Agent 1 and Agent 3 inside the LangGraph workflow.
"""
import json
import logging
import re
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import ValidationError

from app.schemas.output_schema import MethodDetectionOutput
from app.services.llm_factory import get_cached_llm

logger = logging.getLogger(__name__)

# ── Load system prompt from disk (plain text, no f-string interpolation) ──────
_PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "method_detection.txt"
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


def method_detection_agent(state: dict) -> dict:
    """
    LangGraph node: detects the mathematical method used by the student.
    Retries up to MAX_RETRIES times on invalid JSON or schema violations.
    """
    question_text: str = state["question_text"]
    student_steps: list = state["student_steps"]

    llm = get_cached_llm()

    human_text = (
        f"Question:\n{question_text}\n\n"
        f"Student Steps:\n{json.dumps(student_steps, indent=2)}"
    )

    last_error: Exception | None = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            logger.info("[MethodDetectionAgent] Attempt %d/%d", attempt, MAX_RETRIES)
            messages = [
                SystemMessage(content=_SYSTEM_PROMPT),
                HumanMessage(content=human_text),
            ]
            response = llm.invoke(messages)
            raw = _extract_json(response.content)
            validated = MethodDetectionOutput(**raw)
            logger.info(
                "[MethodDetectionAgent] Success — method=%s, confidence=%.2f",
                validated.detected_method,
                validated.confidence,
            )
            return {"method_detection_output": validated.model_dump()}

        except (json.JSONDecodeError, ValidationError, ValueError) as exc:
            last_error = exc
            logger.warning(
                "[MethodDetectionAgent] Attempt %d failed: %s", attempt, exc
            )

    # Fallback: return undetermined method
    logger.error(
        "[MethodDetectionAgent] All %d attempts failed. Using fallback. Last error: %s",
        MAX_RETRIES,
        last_error,
    )
    fallback = MethodDetectionOutput(
        detected_method="undetermined",
        method_is_valid=False,
        alternative_methods_possible=False,
        alternative_methods=[],
        confidence=0.0,
    )
    return {"method_detection_output": fallback.model_dump()}
