"""Mirrors of feedback-service contracts (Module 03).

Source: services/feedback-service/app/models/schemas.py.

The REQUEST models are strict on purpose: the reasoning -> feedback adapter is
the riskiest code in the gateway, and building a strict FeedbackRequest turns a
mapping mistake into a precise local Pydantic error instead of an opaque remote
422 two seconds later.
"""
from typing import List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

# Same model the reasoning request uses — see schemas/common.py.
from app.schemas.common import MarkingScheme, SchemeStep

__all__ = [
    "MarkingScheme",
    "SchemeStep",
    "StepValidity",
    "StepResult",
    "FeedbackRequest",
    "StepFeedback",
    "FeedbackResponse",
]

# feedback-service's StepValidity enum. NOTE this is a different set of values
# from reasoning-service's `status`, which also allows "partially_correct" and
# "unclear". app/services/adapters.py owns the translation.
StepValidity = Literal["correct", "partial", "incorrect"]


class StepResult(BaseModel):
    step_number: int
    expression: str
    validity: StepValidity
    error_description: Optional[str] = None
    marks_awarded: float


class FeedbackRequest(BaseModel):
    question_text: str
    student_steps: List[StepResult]
    detected_method: str
    assigned_marks: float
    marking_scheme: MarkingScheme


class StepFeedback(BaseModel):
    model_config = ConfigDict(extra="ignore")

    step_number: int
    expression: str = ""
    validity: StepValidity = "incorrect"
    marks_awarded: float = 0.0
    what_is_correct: str = ""
    what_is_missing: Optional[str] = None
    why_marks_reduced: Optional[str] = None
    how_to_improve: str = ""
    feedback: str = ""


class FeedbackResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    final_score: float = 0.0
    total_marks: float = 0.0
    step_feedback: List[StepFeedback] = Field(default_factory=list)
    overall_feedback: str = ""
    improvement_suggestions: List[str] = Field(default_factory=list)
