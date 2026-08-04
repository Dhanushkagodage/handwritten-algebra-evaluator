"""Shared multipart form handling for the sync and async entry points."""
from typing import List, Optional

from fastapi import File, Form, UploadFile

from app.config import settings
from app.core.errors import InvalidInputError, PayloadTooLargeError
from app.services.pipeline import PipelineInput, UploadedImage


async def read_upload(upload: UploadFile, *, label: str) -> UploadedImage:
    """Read an upload eagerly.

    This must happen inside the request handler: UploadFile wraps a
    SpooledTemporaryFile that is closed once the request completes, so a
    background job reading it later would fail.
    """
    content = await upload.read()
    if not content:
        raise InvalidInputError(f"{label} is empty.")
    if len(content) > settings.max_upload_bytes:
        raise PayloadTooLargeError(
            f"{label} is {len(content) / 1_048_576:.1f} MB; the limit is "
            f"{settings.max_upload_mb} MB."
        )
    return (
        upload.filename or "upload.jpg",
        content,
        upload.content_type or "image/jpeg",
    )


async def build_pipeline_input(
    *,
    run_id: str,
    answer_images: List[UploadFile],
    marking_scheme_image: UploadFile,
    question_text: str,
    ocr_mode: Optional[str],
    use_math_ocr: Optional[bool],
    question_id: Optional[str],
    multi_question_policy: Optional[str],
) -> PipelineInput:
    if not answer_images:
        raise InvalidInputError("Upload at least one answer image.")
    if len(answer_images) > settings.max_answer_images:
        raise InvalidInputError(
            f"Upload at most {settings.max_answer_images} answer images "
            f"(received {len(answer_images)})."
        )

    images = [
        await read_upload(upload, label=f"Answer image {index}")
        for index, upload in enumerate(answer_images, start=1)
    ]
    scheme = await read_upload(marking_scheme_image, label="The marking scheme image")

    return PipelineInput(
        answer_images=images,
        marking_scheme_image=scheme,
        question_text=question_text or "",
        ocr_mode=ocr_mode or None,
        use_math_ocr=use_math_ocr,
        question_id=(question_id or "").strip() or None,
        multi_question_policy=(multi_question_policy or "").strip() or None,
        run_id=run_id,
    )


# ── Reusable Form/File declarations (identical across both endpoints) ────────

AnswerImages = File(..., description="One to five ordered images of the student's answer.")
SchemeImage = File(..., description="An image of the marking scheme for this question.")
QuestionText = Form("", description="Optional typed question text; improves OCR accuracy.")
OcrMode = Form(None, description="'openai_vision' (default) or 'local'.")
UseMathOcr = Form(None, description="Enable the slower pix2tex math OCR in local mode.")
QuestionId = Form(None, description="Which detected question to grade, e.g. Q2.")
MultiQuestionPolicy = Form(None, description="'first' (default), 'all' or 'error'.")
