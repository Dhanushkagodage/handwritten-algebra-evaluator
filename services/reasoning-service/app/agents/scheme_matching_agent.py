"""
Agent 3: Scheme Matching Agent
──────────────────────────────
INPUT : student_steps + marking_scheme  (DOES see marking scheme)
OUTPUT: SchemeMatchingOutput — step-to-scheme mappings with match scores

Runs in parallel with Agent 1 and Agent 2 inside the LangGraph workflow.
"""
import json
import logging
import re
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import ValidationError

from app.schemas.output_schema import SchemeMatchingOutput
from app.services.llm_factory import get_cached_llm

logger = logging.getLogger(__name__)

# ── Load system prompt from disk (plain text, no f-string interpolation) ──────
_PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "scheme_matching.txt"
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


def scheme_matching_agent(state: dict) -> dict:
    """
    LangGraph node: maps student steps to marking scheme steps.
    Retries up to MAX_RETRIES times on invalid JSON or schema violations.
    """
    student_steps: list = state["student_steps"]
    marking_scheme: dict = state["marking_scheme"]

    llm = get_cached_llm()

    human_text = (
        f"Student Steps:\n{json.dumps(student_steps, indent=2)}\n\n"
        f"Marking Scheme Steps:\n{json.dumps(marking_scheme['steps'], indent=2)}"
    )

    last_error: Exception | None = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            logger.info("[SchemeMatchingAgent] Attempt %d/%d", attempt, MAX_RETRIES)
            messages = [
                SystemMessage(content=_SYSTEM_PROMPT),
                HumanMessage(content=human_text),
            ]
            response = llm.invoke(messages)
            raw = _extract_json(response.content)
            validated = SchemeMatchingOutput(**raw)
            logger.info(
                "[SchemeMatchingAgent] Success — %d step matches, %d unmatched scheme steps",
                len(validated.step_matches),
                len(validated.unmatched_scheme_steps),
            )
            return {"scheme_matching_output": validated.model_dump()}

        except (json.JSONDecodeError, ValidationError, ValueError) as exc:
            last_error = exc
            logger.warning(
                "[SchemeMatchingAgent] Attempt %d failed: %s", attempt, exc
            )

    # Fallback: no matches
    logger.error(
        "[SchemeMatchingAgent] All %d attempts failed. Using fallback. Last error: %s",
        MAX_RETRIES,
        last_error,
    )
    fallback = SchemeMatchingOutput(
        step_matches=[
            {
                "step_id": s["step_id"],
                "matched_scheme_step": None,
                "match_score": 0.0,
                "is_partial_match": False,
                "missing_elements": ["Agent evaluation failed"],
            }
            for s in student_steps
        ],
        unmatched_scheme_steps=[s["step_no"] for s in marking_scheme["steps"]],
    )
    return {"scheme_matching_output": fallback.model_dump()}
