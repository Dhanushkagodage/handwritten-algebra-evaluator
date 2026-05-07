from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, model_validator


class StepValidity(str, Enum):
    CORRECT = "correct"
    PARTIAL = "partial"
    INCORRECT = "incorrect"


class StepResult(BaseModel):
    step_number: int
    expression: str
    # Primary field — set by Module 02 when it outputs step_validity
    validity: Optional[StepValidity] = None
    # Kept for backward compatibility with current Module 02 output (is_correct: bool)
    is_correct: Optional[bool] = None
    error_description: Optional[str] = None
    marks_awarded: float

    @model_validator(mode="after")
    def derive_validity(self):
        if self.validity is None and self.is_correct is not None:
            self.validity = (
                StepValidity.CORRECT if self.is_correct else StepValidity.INCORRECT
            )
        if self.validity is None:
            self.validity = StepValidity.INCORRECT
        return self


class MarkingSchemeStep(BaseModel):
    step_number: int
    expected_expression: str
    marks: float
    description: Optional[str] = None


class FeedbackRequest(BaseModel):
    question_text: str
    student_steps: List[StepResult]
    detected_method: str
    assigned_marks: float   # total score from Module 02
    total_marks: float
    marking_scheme: List[MarkingSchemeStep]


class StepFeedback(BaseModel):
    step_number: int
    expression: str
    validity: StepValidity
    marks_awarded: float
    # Four-component feedback structure (per project spec)
    what_is_correct: str
    what_is_missing: Optional[str] = None    # only for INCORRECT / PARTIAL
    why_marks_reduced: Optional[str] = None  # only for INCORRECT / PARTIAL
    how_to_improve: str
    feedback: str                            # combined one-line summary


class FeedbackResponse(BaseModel):
    final_score: float
    total_marks: float
    step_feedback: List[StepFeedback]
    overall_feedback: str
    improvement_suggestions: List[str]
