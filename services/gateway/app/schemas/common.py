"""The marking scheme — the one contract all three modules already agree on.

ocr-service emits it, reasoning-service consumes it, and feedback-service
consumes it again, with identical field names and types in each. Defining it
once here means the gateway can hand the same object to both downstream calls;
declaring it separately per service made passing it along a type error.
"""
from typing import List

from pydantic import BaseModel, Field


class SchemeStep(BaseModel):
    step_no: int
    description: str
    expected_expression: str
    marks: float = Field(..., ge=0)


class MarkingScheme(BaseModel):
    #: Marks AVAILABLE. Note reasoning-service's EvaluationOutput.total_marks
    #: means marks EARNED — same name, opposite meaning.
    total_marks: float = Field(..., ge=0)
    steps: List[SchemeStep]
