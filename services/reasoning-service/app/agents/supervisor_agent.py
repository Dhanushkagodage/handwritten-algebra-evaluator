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
from collections import defaultdict
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import ValidationError

from app.schemas.output_schema import EvaluationOutput, StepAnalysis
from app.services.llm_factory import get_cached_llm
from app.config.settings import settings

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


def _j(data) -> str:
    """Compact JSON serialisation — minimal tokens for LLM input."""
    return json.dumps(data, separators=(",", ":"), ensure_ascii=False)


def _apply_marking_rules(
    output: EvaluationOutput,
    step_validation: dict,
    scheme_mark_map: dict,
    method_detection: dict | None = None,
) -> EvaluationOutput:
    """
    Deterministic post-processor applied after BOTH the LLM path and the fallback path.

    Rule 1 — Incorrect steps always get 0 marks, UNLESS the supervisor LLM independently
              awarded marks > 0 to that step (coherence override: supervisor verified the
              step is actually correct despite step_validation flagging it).
    Rule 2 — When multiple student steps map to the same scheme step (sub-steps),
              only the single best-matching step earns marks for that scheme step;
              all other sub-steps are zeroed. This prevents total_marks > max_marks.
    Rule 3 — Alternative Method Consistency: when the student used a valid alternative
              method (method_is_valid=True, alternative_methods_possible=True) AND all
              student steps are mathematically correct (no 'incorrect' status), any
              remaining unawarded scheme marks are redistributed proportionally across
              correct steps that were left at 0 due to null scheme matching. This ensures
              a fully-correct alternative-method solution always earns max_marks.
    """
    steps: list[StepAnalysis] = output.steps_analysis

    # Build quick status lookup from step-validation agent output
    status_map: dict[int, str] = {
        v["step_id"]: v.get("status", "unclear")
        for v in step_validation.get("step_validations", [])
    }

    # Rule 1: force 0 marks for any step the validation agent marked incorrect,
    # BUT respect coherence override: if the supervisor LLM already awarded marks > 0
    # for this step, it means the supervisor independently verified the step is correct
    # (e.g., the student wrote an unevaluated-but-valid intermediate form).
    for step in steps:
        if status_map.get(step.step_id) == "incorrect":
            if step.marks_awarded > 0:
                # Supervisor says correct (marks > 0) but step_validation says incorrect.
                # Trust the supervisor — it has full context and performs its own math check.
                logger.info(
                    "[SupervisorAgent] CoherenceOverride: step %d kept %.1f marks "
                    "(supervisor verified correct; step_validation was wrong).",
                    step.step_id,
                    step.marks_awarded,
                )
                # Align the validity fields to match supervisor's verdict
                step.validity = True
                step.status = "correct"
            else:
                # Both supervisor and step_validation agree it's wrong → zero marks
                logger.info(
                    "[SupervisorAgent] Rule1: step %d set to 0 (incorrect)", step.step_id
                )
                step.marks_awarded = 0.0
                step.validity = False
                step.status = "incorrect"

    # Rule 2: group sub-steps by scheme step; keep marks only on the best one
    groups: dict[int, list[StepAnalysis]] = defaultdict(list)
    for step in steps:
        if step.matched_scheme_step is not None:
            groups[step.matched_scheme_step].append(step)

    for scheme_no, group in groups.items():
        if len(group) == 1:
            continue  # single mapping — no issue
        cap = float(scheme_mark_map.get(scheme_no, 0.0))
        total = sum(s.marks_awarded for s in group)
        if total <= cap:
            continue  # already within limit — no issue

        # Pick the step with the highest match_score (last step wins ties)
        best = max(group, key=lambda s: (s.match_score, s.step_id))
        best_marks = min(round(best.match_score * cap * 2) / 2, cap)

        logger.info(
            "[SupervisorAgent] Rule2: scheme step %d — %d sub-steps, total=%.1f > cap=%.1f. "
            "Awarding %.1f to step %d only.",
            scheme_no, len(group), total, cap, best_marks, best.step_id,
        )
        for step in group:
            step.marks_awarded = best_marks if step is best else 0.0

    # ── Rule 3: Alternative Method Consistency ────────────────────────────────
    # When method_detection confirms a valid alternative method AND every student
    # step is correct, redistribute any unawarded scheme marks to correct steps
    # that were left at 0 only because the scheme matcher returned null.
    if method_detection is not None:
        is_alt = (
            method_detection.get("method_is_valid", False)
            and method_detection.get("alternative_methods_possible", False)
        )
        if is_alt:
            # Check all steps are correct (none marked incorrect)
            all_correct = all(
                status_map.get(s.step_id, "unclear") != "incorrect"
                for s in steps
            )
            if all_correct:
                awarded_so_far = round(sum(s.marks_awarded for s in steps), 2)
                unawarded = round(output.max_marks - awarded_so_far, 2)

                if unawarded > 0:
                    # Steps that are correct but got 0 marks (unmatched by scheme matcher)
                    zero_correct_steps = [
                        s for s in steps
                        if s.marks_awarded == 0.0
                        and status_map.get(s.step_id, "unclear") in ("correct", "partially_correct")
                    ]
                    if zero_correct_steps:
                        # Distribute unawarded marks equally across these steps
                        share = round(unawarded / len(zero_correct_steps) * 2) / 2  # round to 0.5
                        # Clamp total distribution to not exceed max_marks
                        total_share = share * len(zero_correct_steps)
                        if awarded_so_far + total_share > output.max_marks:
                            share = round(
                                (unawarded / len(zero_correct_steps)) * 2
                            ) / 2
                        logger.info(
                            "[SupervisorAgent] Rule3 (AltMethod): redistributing %.1f unawarded "
                            "marks across %d zero-marked correct steps (%.1f each).",
                            unawarded, len(zero_correct_steps), share,
                        )
                        for s in zero_correct_steps:
                            s.marks_awarded = share
                            s.validity = True
                            s.status = "correct"

    # Recompute totals
    output.total_marks = round(sum(s.marks_awarded for s in steps), 2)
    # Clamp to max_marks to prevent floating-point overshoot
    output.total_marks = min(output.total_marks, output.max_marks)
    output.percentage = (
        round((output.total_marks / output.max_marks) * 100, 1)
        if output.max_marks > 0 else 0.0
    )
    return output



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

    max_marks = float(marking_scheme.get("total_marks", 0.0))
    total_marks = sum(s["marks_awarded"] for s in steps_analysis)
    percentage = round((total_marks / max_marks) * 100, 1) if max_marks > 0 else 0.0

    fallback = EvaluationOutput(
        steps_analysis=steps_analysis,
        total_marks=total_marks,
        max_marks=max_marks,
        percentage=percentage,
        summary="Evaluation completed using fallback logic (supervisor LLM unavailable).",
        method_feedback=f"Detected method: {method_detection.get('detected_method', 'undetermined')}.",
        missing_steps_feedback=None,
    )
    return _apply_marking_rules(fallback, step_validation, scheme_mark_map, method_detection)


async def supervisor_agent(state: dict) -> dict:
    """
    LangGraph node: synthesizes all agent outputs into a final evaluation.
    Falls back to deterministic mark computation if LLM fails MAX_RETRIES times.
    """
    step_validation: dict = state["step_validation_output"]
    method_detection: dict = state["method_detection_output"]
    scheme_matching: dict = state["scheme_matching_output"]
    marking_scheme: dict = state["marking_scheme"]

    scheme_mark_map: dict = {
        s["step_no"]: s["marks"] for s in marking_scheme["steps"]
    }

    llm = get_cached_llm(
        provider=settings.llm_provider,
        model=settings.get_supervisor_model(),
        temperature=settings.llm_temperature,
    )

    human_text = (
        f"## Step Validation Agent Output\n{_j(step_validation)}\n\n"
        f"## Method Detection Agent Output\n{_j(method_detection)}\n\n"
        f"## Scheme Matching Agent Output\n{_j(scheme_matching)}\n\n"
        f"## Official Marking Scheme\n{_j(marking_scheme)}"
    )

    last_error: Exception | None = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            logger.info("[SupervisorAgent] Attempt %d/%d", attempt, MAX_RETRIES)
            messages = [
                SystemMessage(content=_SYSTEM_PROMPT),
                HumanMessage(content=human_text),
            ]
            response = await llm.ainvoke(messages)
            raw = _extract_json(response.content)
            validated = EvaluationOutput(**raw)
            validated = _apply_marking_rules(validated, step_validation, scheme_mark_map, method_detection)
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
