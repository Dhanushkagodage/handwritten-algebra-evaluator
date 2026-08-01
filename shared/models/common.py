from pydantic import BaseModel, Field
from typing import List, Optional


class StudentStep(BaseModel):
    step_number: int
    expression: str
    is_correct: Optional[bool] = None
    error_description: Optional[str] = None
    marks_awarded: float = 0.0


class SchemeStep(BaseModel):
    """One marking scheme step. All four fields are required."""
    step_no: int
    description: str
    expected_expression: str
    marks: float = Field(..., ge=0)


class MarkingScheme(BaseModel):
    """The canonical marking scheme exchanged between all three modules.
    `total_marks` lives here and is never duplicated alongside the scheme."""
    total_marks: float = Field(..., ge=0)
    steps: List[SchemeStep]


class OCROutput(BaseModel):
    question_text: str
    answer_text: str
    student_steps: List[StudentStep]
    linked_question: str


class ReasoningResult(BaseModel):
    question_text: str
    student_steps: List[StudentStep]
    detected_method: str
    assigned_marks: float
    marking_scheme: MarkingScheme  # .total_marks is the single source of truth


class FeedbackOutput(BaseModel):
    final_score: float
    total_marks: float
    step_feedback: List[dict]
    overall_feedback: str
    improvement_suggestions: List[str]
