from pydantic import BaseModel
from typing import List

class Step(BaseModel):
    step: int
    content: str

class StepRequest(BaseModel):
    question: str
    student_answer: List[Step]