"""
Output schemas for all agents in the multi-agent evaluation system.
Strict Pydantic models ensure structured, validated LLM outputs.
"""
from pydantic import BaseModel, Field
from typing import List, Optional


# ─── Agent 1: Step Validation Agent Output ────────────────────────────────────

class StepValidationResult(BaseModel):
    step_id: int = Field(..., description="Reference to student step ID")
    is_valid: bool = Field(..., description="Whether this step is mathematically valid")
    status: str = Field(
        ...,
        description="One of: correct | incorrect | partially_correct | unclear"
    )
    error: Optional[str] = Field(None, description="Description of error if step is invalid")
    confidence: float = Field(
        ..., ge=0.0, le=1.0,
        description="Agent confidence in this validation (0–1)"
    )


class StepValidationOutput(BaseModel):
    step_validations: List[StepValidationResult]
    missing_transitions: List[str] = Field(
        default_factory=list,
        description="List of detected missing logical transitions between steps"
    )


# ─── Agent 2: Method Detection Agent Output ───────────────────────────────────

class MethodDetectionOutput(BaseModel):
    detected_method: str = Field(
        ..., description="Primary solving method detected (e.g., factorization, substitution)"
    )
    method_is_valid: bool = Field(
        ..., description="Whether the detected method is mathematically appropriate for this question"
    )
    alternative_methods_possible: bool = Field(
        ..., description="Whether alternative valid methods exist for this question"
    )
    alternative_methods: List[str] = Field(
        default_factory=list,
        description="List of alternative valid methods if applicable"
    )
    confidence: float = Field(
        ..., ge=0.0, le=1.0,
        description="Agent confidence in method detection (0–1)"
    )


# ─── Agent 3: Scheme Matching Agent Output ────────────────────────────────────

class StepMatchResult(BaseModel):
    step_id: int = Field(..., description="Student step ID")
    matched_scheme_step: Optional[int] = Field(
        None, description="Matched marking scheme step number (null if no match)"
    )
    match_score: float = Field(
        ..., ge=0.0, le=1.0,
        description="Similarity score between student step and scheme step"
    )
    is_partial_match: bool = Field(
        ..., description="Whether this is a partial rather than full match"
    )
    missing_elements: List[str] = Field(
        default_factory=list,
        description="Elements required by scheme step that are missing from student step"
    )


class SchemeMatchingOutput(BaseModel):
    step_matches: List[StepMatchResult]
    unmatched_scheme_steps: List[int] = Field(
        default_factory=list,
        description="Marking scheme step numbers with no corresponding student step"
    )


# ─── Supervisor Agent Output ──────────────────────────────────────────────────

class StepAnalysis(BaseModel):
    step_id: int
    validity: bool
    status: str
    method: str
    matched_scheme_step: Optional[int]
    match_score: float
    marks_awarded: float
    max_marks: float
    reason: str
    confidence: float


class EvaluationOutput(BaseModel):
    steps_analysis: List[StepAnalysis]
    total_marks: float
    max_marks: float
    percentage: float
    summary: str
    method_feedback: str
    missing_steps_feedback: Optional[str] = None

    # ── Debug / Intermediate Outputs ──────────────────────────────────────────
    step_validation: Optional[StepValidationOutput] = Field(
        None, description="Detailed intermediate output from Step Validation Agent"
    )
    method_detection: Optional[MethodDetectionOutput] = Field(
        None, description="Detailed intermediate output from Method Detection Agent"
    )
    scheme_matching: Optional[SchemeMatchingOutput] = Field(
        None, description="Detailed intermediate output from Scheme Matching Agent"
    )

