"""Mirrors of ocr-service responses (Module 01).

Source: services/ocr-service/src/openai_vision_ocr.py (parse_reasoning_json,
parse_marking_scheme_json) and src/structure_output.py (build_reasoning_input).

ocr-service declares no Pydantic models at all — its output shape is only as
stable as the vision prompt. These models are therefore deliberately LENIENT:
unknown keys are ignored and everything non-essential has a default, so a
prompt tweak upstream degrades into a warning rather than a 502.
"""
from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator


class OcrStudentStep(BaseModel):
    model_config = ConfigDict(extra="ignore")

    step_id: int
    content: str = ""


class OcrReasoningInput(BaseModel):
    model_config = ConfigDict(extra="ignore")

    question_id: Optional[str] = None
    question_text: str = ""
    student_steps: List[OcrStudentStep] = Field(default_factory=list)
    final_answer: Optional[str] = None


class OcrExtractResponse(BaseModel):
    """`/extract` and `/extract-pages` return exactly one of these two keys.

    Singular when the vision model found one question, plural (with question_id
    on each entry) when it found several.
    """

    model_config = ConfigDict(extra="ignore")

    reasoning_input: Optional[OcrReasoningInput] = None
    reasoning_inputs: Optional[List[OcrReasoningInput]] = None

    @model_validator(mode="after")
    def _require_one(self):
        if self.reasoning_input is None and self.reasoning_inputs is None:
            raise ValueError(
                "OCR response contained neither 'reasoning_input' nor 'reasoning_inputs'."
            )
        return self


class OcrSchemeStep(BaseModel):
    model_config = ConfigDict(extra="ignore")

    step_no: int
    description: str = ""
    expected_expression: str = ""
    marks: float = 0.0


class OcrMarkingScheme(BaseModel):
    model_config = ConfigDict(extra="ignore")

    total_marks: float = 0.0
    steps: List[OcrSchemeStep] = Field(default_factory=list)


class OcrMarkingSchemeResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    marking_scheme: OcrMarkingScheme
