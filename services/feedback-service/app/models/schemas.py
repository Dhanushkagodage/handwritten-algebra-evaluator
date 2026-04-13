from pydantic import BaseModel
from typing import List, Optional


class StepResult(BaseModel):
    step_number: int
    expression: str
    is_correct: bool
    error_description: Optional[str] = None
    marks_awarded: float


class MarkingSchemeStep(BaseModel):
    step_number: int
    expected_expression: str
    marks: float
    description: Optional[str] = None


class FeedbackRequest(BaseModel):
    question_text: str
    student_steps: List[StepResult]
    detected_method: str
    assigned_marks: float
    total_marks: float
    marking_scheme: List[MarkingSchemeStep]


class StepFeedback(BaseModel):
    step_number: int
    expression: str
    is_correct: bool
    marks_awarded: float
    feedback: str


class FeedbackResponse(BaseModel):
    final_score: float
    total_marks: float
    step_feedback: List[StepFeedback]
    overall_feedback: str
    improvement_suggestions: List[str]
