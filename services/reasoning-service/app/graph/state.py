"""
LangGraph state definition for the multi-agent evaluation workflow.
Uses TypedDict with Optional fields to cleanly represent pre/post agent state.
"""
from typing import Any, Dict, List, Optional, TypedDict


class EvaluationState(TypedDict, total=False):
    # ── Input A: reasoning inputs (shared by Agent 1 & 2) ─────────────────────
    question_text: str
    student_steps: List[Dict[str, Any]]
    final_answer: Optional[str]

    # ── Input B: marking scheme (used by Agent 3 & Supervisor) ────────────────
    marking_scheme: Dict[str, Any]

    # ── Agent outputs (populated after parallel execution) ────────────────────
    step_validation_output: Dict[str, Any]
    method_detection_output: Dict[str, Any]
    scheme_matching_output: Dict[str, Any]

    # ── Final supervisor output ───────────────────────────────────────────────
    evaluation_output: Dict[str, Any]