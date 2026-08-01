"""POST /api/v1/evaluate — the synchronous one-call pipeline."""
import uuid
from typing import List, Optional

from fastapi import APIRouter, Request, UploadFile

from app.api import uploads
from app.schemas.gateway import EvaluationResult
from app.services.pipeline import run_pipeline

router = APIRouter(prefix="/api/v1", tags=["Evaluation"])


@router.post(
    "/evaluate",
    response_model=EvaluationResult,
    summary="Run the full OCR -> reasoning -> feedback pipeline (synchronous)",
    description=(
        "Runs all three modules in one call and returns the graded, explained result.\n\n"
        "**This request stays open for the whole pipeline — typically 60-90 seconds, and "
        "up to 3-4 minutes when the feedback model's Hugging Face Space is cold.** It "
        "exists for command-line use and testing. Browsers should use POST /api/v1/jobs "
        "and poll instead, which also reports which stage is running."
    ),
)
async def evaluate(
    request: Request,
    answer_images: List[UploadFile] = uploads.AnswerImages,
    marking_scheme_image: UploadFile = uploads.SchemeImage,
    question_text: str = uploads.QuestionText,
    ocr_mode: Optional[str] = uploads.OcrMode,
    use_math_ocr: Optional[bool] = uploads.UseMathOcr,
    question_id: Optional[str] = uploads.QuestionId,
    multi_question_policy: Optional[str] = uploads.MultiQuestionPolicy,
) -> EvaluationResult:
    payload = await uploads.build_pipeline_input(
        run_id=uuid.uuid4().hex[:12],
        answer_images=answer_images,
        marking_scheme_image=marking_scheme_image,
        question_text=question_text,
        ocr_mode=ocr_mode,
        use_math_ocr=use_math_ocr,
        question_id=question_id,
        multi_question_policy=multi_question_policy,
    )
    output = await run_pipeline(request.app.state.http, payload)
    return output.result
