from pydantic import BaseModel
from typing import List, Optional


class StudentStep(BaseModel):
    step_number: int
    expression: str
    is_correct: Optional[bool] = None
    error_description: Optional[str] = None
    marks_awarded: float = 0.0


class MarkingSchemeStep(BaseModel):
    step_number: int
    expected_expression: str
    marks: float
    description: Optional[str] = None


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
    total_marks: float
    marking_scheme: List[MarkingSchemeStep]


class FeedbackOutput(BaseModel):
    final_score: float
    total_marks: float
    step_feedback: List[dict]
    overall_feedback: str
    improvement_suggestions: List[str]
