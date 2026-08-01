"""
Input schemas for the algebra evaluation multi-agent system.
Separates Input A (reasoning) from Input B (marking scheme) per design spec.
"""
from pydantic import BaseModel, Field
from typing import List, Optional


# ─── Input A: Reasoning Input ─────────────────────────────────────────────────

class StudentStep(BaseModel):
    step_id: int = Field(..., description="Unique identifier for this step")
    content: str = Field(..., description="Mathematical content/expression for this step")


class ReasoningInput(BaseModel):
    question_text: str = Field(..., description="The original algebra question")
    student_steps: List[StudentStep] = Field(..., description="Ordered list of student solution steps")
    final_answer: Optional[str] = Field(None, description="Student's stated final answer")


# ─── Input B: Marking Scheme ──────────────────────────────────────────────────

class SchemeStep(BaseModel):
    step_no: int = Field(..., description="Step number in the marking scheme")
    description: str = Field(..., description="Description of what this step requires")
    expected_expression: str = Field(..., description="The algebraic result this step should produce")
    marks: float = Field(..., ge=0, description="Marks awarded for this scheme step")


class MarkingScheme(BaseModel):
    total_marks: float = Field(..., ge=0, description="Maximum marks for this question")
    steps: List[SchemeStep] = Field(..., description="Ordered marking scheme steps")


# ─── Combined Evaluation Request ──────────────────────────────────────────────

class EvaluationRequest(BaseModel):
    reasoning_input: ReasoningInput = Field(..., description="Student answer and question (Input A)")
    marking_scheme: MarkingScheme = Field(..., description="Marking scheme (Input B)")

    class Config:
        json_schema_extra = {
            "example": {
                "reasoning_input": {
                    "question_text": "Solve the quadratic equation: x² - 5x + 6 = 0",
                    "student_steps": [
                        {"step_id": 1, "content": "x² - 5x + 6 = 0"},
                        {"step_id": 2, "content": "(x - 2)(x - 3) = 0"},
                        {"step_id": 3, "content": "x = 2 or x = 3"}
                    ],
                    "final_answer": "x = 2 or x = 3"
                },
                "marking_scheme": {
                    "total_marks": 5,
                    "steps": [
                        {"step_no": 1, "description": "Correctly write the standard form", "expected_expression": "x² - 5x + 6 = 0", "marks": 1},
                        {"step_no": 2, "description": "Correctly factorise the quadratic", "expected_expression": "(x - 2)(x - 3) = 0", "marks": 2},
                        {"step_no": 3, "description": "State both correct solutions", "expected_expression": "x = 2 or x = 3", "marks": 2}
                    ]
                }
            }
        }
