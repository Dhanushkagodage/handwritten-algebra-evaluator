"""Mirrors of reasoning-service contracts (Module 02).

Source: services/reasoning-service/app/schemas/input_schema.py and
output_schema.py. Mirrored rather than imported — reasoning-service pulls in
LangGraph/LangChain, which the gateway deliberately does not depend on.

Request models are STRICT (a bad request should fail here with a clear message,
not as a remote 422). Response models are LENIENT — the supervisor agent can be
an LLM, and an unexpected extra field should not take the pipeline down.
"""
from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field

# The marking scheme is identical across all three services — see schemas/common.py.
from app.schemas.common import MarkingScheme, SchemeStep

__all__ = [
    "MarkingScheme",
    "SchemeStep",
    "StudentStep",
    "ReasoningInput",
    "EvaluationRequest",
    "StepValidationResult",
    "StepValidationOutput",
    "MethodDetectionOutput",
    "StepMatchResult",
    "SchemeMatchingOutput",
    "StepAnalysis",
    "EvaluationOutput",
]


# ── Request side (strict) ────────────────────────────────────────────────────

class StudentStep(BaseModel):
    step_id: int
    content: str


class ReasoningInput(BaseModel):
    question_text: str
    student_steps: List[StudentStep]
    final_answer: Optional[str] = None


class EvaluationRequest(BaseModel):
    reasoning_input: ReasoningInput
    marking_scheme: MarkingScheme


# ── Response side (lenient) ──────────────────────────────────────────────────

class StepValidationResult(BaseModel):
    model_config = ConfigDict(extra="ignore")

    step_id: int
    is_valid: Optional[bool] = None
    status: Optional[str] = None
    error: Optional[str] = None
    confidence: Optional[float] = None


class StepValidationOutput(BaseModel):
    model_config = ConfigDict(extra="ignore")

    step_validations: List[StepValidationResult] = Field(default_factory=list)
    missing_transitions: List[str] = Field(default_factory=list)


class MethodDetectionOutput(BaseModel):
    model_config = ConfigDict(extra="ignore")

    detected_method: str = ""
    method_is_valid: Optional[bool] = None
    alternative_methods_possible: Optional[bool] = None
    alternative_methods: List[str] = Field(default_factory=list)
    confidence: Optional[float] = None


class StepMatchResult(BaseModel):
    model_config = ConfigDict(extra="ignore")

    step_id: int
    matched_scheme_step: Optional[int] = None
    match_score: Optional[float] = None
    is_partial_match: Optional[bool] = None
    missing_elements: List[str] = Field(default_factory=list)


class SchemeMatchingOutput(BaseModel):
    model_config = ConfigDict(extra="ignore")

    step_matches: List[StepMatchResult] = Field(default_factory=list)
    unmatched_scheme_steps: List[int] = Field(default_factory=list)


class StepAnalysis(BaseModel):
    model_config = ConfigDict(extra="ignore")

    step_id: int
    # NOTE: a BOOLEAN here. feedback-service's StepFeedback.validity is a
    # three-value STRING with the same name. Never pass this through.
    validity: Optional[bool] = None
    status: Optional[str] = None
    method: Optional[str] = None
    matched_scheme_step: Optional[int] = None
    match_score: Optional[float] = None
    marks_awarded: float = 0.0
    max_marks: float = 0.0
    confidence: Optional[float] = None


class EvaluationOutput(BaseModel):
    model_config = ConfigDict(extra="ignore")

    steps_analysis: List[StepAnalysis] = Field(default_factory=list)
    total_marks: float = 0.0      # marks EARNED
    max_marks: float = 0.0        # marks AVAILABLE
    percentage: float = 0.0
    summary: str = ""
    method_feedback: str = ""
    missing_steps_feedback: Optional[str] = None
    step_validation: Optional[StepValidationOutput] = None
    method_detection: Optional[MethodDetectionOutput] = None
    scheme_matching: Optional[SchemeMatchingOutput] = None
