"""The gateway's own public contract — what the frontend actually consumes."""
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

from app.schemas.upstream_feedback import FeedbackResponse, StepFeedback
from app.schemas.upstream_ocr import OcrStudentStep
from app.schemas.upstream_reasoning import EvaluationOutput, MarkingScheme

# The three middle values match PIPELINE_STEPS in frontend/src/pages/Evaluate.tsx,
# so the existing progress tracker can be driven straight off `stage`.
StageKey = Literal["queued", "ocr", "reasoning", "feedback", "done"]
StageStatus = Literal["pending", "running", "succeeded", "failed", "skipped"]
JobStatus = Literal["pending", "running", "succeeded", "failed", "cancelled"]


class StageRecord(BaseModel):
    key: StageKey
    status: StageStatus = "pending"
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    duration_ms: Optional[int] = None
    message: Optional[str] = None


class ErrorEnvelope(BaseModel):
    error: bool = True
    error_code: str
    message: str
    # `detail` mirrors `message` so frontend code reading err.response.data.detail
    # (the FastAPI convention) keeps working unchanged.
    detail: str
    stage: Optional[str] = None
    status_code: int = 500
    details: Dict[str, Any] = Field(default_factory=dict)


class QuestionResult(BaseModel):
    """Everything the pipeline produced for one question."""

    question_id: str
    # reasoning-service never echoes question_text back, so the gateway is its
    # only holder between the OCR and feedback stages.
    question_text: str
    marking_scheme: MarkingScheme
    student_steps: List[OcrStudentStep]
    final_answer: Optional[str] = None
    reasoning: EvaluationOutput
    feedback: FeedbackResponse


class EvaluationResult(BaseModel):
    """Flattened primary question plus the full per-question detail.

    The flattened block mirrors FeedbackResponse field-for-field so the results
    page can read it directly; `questions` carries everything for a future
    multi-question UI.
    """

    final_score: float
    total_marks: float
    step_feedback: List[StepFeedback]
    overall_feedback: str
    improvement_suggestions: List[str]

    question_id: str
    question_text: str
    question_count: int
    questions: List[QuestionResult]
    warnings: List[str] = Field(default_factory=list)
    timings_ms: Dict[str, int] = Field(default_factory=dict)


class JobCreatedResponse(BaseModel):
    job_id: str
    status: JobStatus
    stage: StageKey
    poll_url: str
    poll_after_ms: int = 1500


class JobStatusResponse(BaseModel):
    job_id: str
    status: JobStatus
    stage: StageKey
    stage_message: Optional[str] = None
    stages: List[StageRecord]
    progress: float = 0.0
    created_at: datetime
    updated_at: datetime
    elapsed_ms: int = 0
    poll_after_ms: int = 1500
    warnings: List[str] = Field(default_factory=list)
    result: Optional[EvaluationResult] = None
    error: Optional[ErrorEnvelope] = None


class ServiceHealth(BaseModel):
    status: Literal["up", "down"]
    url: str
    latency_ms: Optional[int] = None
    details: Dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = None
    note: Optional[str] = None


class ServicesHealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    services: Dict[str, ServiceHealth]
