"""The one call: OCR -> reasoning -> feedback.

`run_pipeline` is shared verbatim by the synchronous endpoint and the background
job runner; only the `on_stage` callback differs.
"""
import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Awaitable, Callable, List, Optional, Tuple

import httpx

from app.clients import feedback as feedback_client
from app.clients import ocr as ocr_client
from app.clients import reasoning as reasoning_client
from app.config import settings
from app.core.errors import InvalidInputError
from app.schemas.gateway import EvaluationResult, QuestionResult
from app.services.adapters import (
    build_evaluation_request,
    build_feedback_request,
    detect_reasoning_fallback,
    normalize_marking_scheme,
    normalize_ocr_questions,
    select_questions,
)

logger = logging.getLogger(__name__)

#: (filename, bytes, content_type)
UploadedImage = Tuple[str, bytes, str]

#: on_stage(stage, message) -> awaitable
StageCallback = Callable[[str, Optional[str]], Awaitable[None]]

FEEDBACK_STAGE_MESSAGE = (
    "Generating stepwise feedback — a cold Hugging Face Space can take up to 90 seconds."
)


@dataclass
class PipelineInput:
    answer_images: List[UploadedImage]
    marking_scheme_image: UploadedImage
    question_text: str = ""
    ocr_mode: Optional[str] = None
    use_math_ocr: Optional[bool] = None
    question_id: Optional[str] = None
    multi_question_policy: Optional[str] = None
    #: Prefix applied to uploaded filenames so concurrent runs cannot collide in
    #: ocr-service's data/raw directory.
    run_id: str = "run"

    def release_images(self) -> None:
        """Drop the image bytes once OCR is done, so long jobs stop holding them."""
        self.answer_images = []
        self.marking_scheme_image = ("", b"", "")


@dataclass
class PipelineOutput:
    result: EvaluationResult
    warnings: List[str] = field(default_factory=list)


def validate_input(payload: PipelineInput) -> None:
    if not payload.answer_images:
        raise InvalidInputError("Upload at least one answer image.")
    if len(payload.answer_images) > settings.max_answer_images:
        raise InvalidInputError(
            f"Upload at most {settings.max_answer_images} answer images "
            f"(received {len(payload.answer_images)})."
        )

    mode = payload.ocr_mode or settings.default_ocr_mode
    if mode not in {"openai_vision", "local"}:
        raise InvalidInputError("ocr_mode must be 'openai_vision' or 'local'.")

    policy = payload.multi_question_policy or settings.multi_question_policy
    if policy not in {"first", "all", "error"}:
        raise InvalidInputError("multi_question_policy must be 'first', 'all' or 'error'.")


async def _noop_stage(stage: str, message: Optional[str] = None) -> None:  # pragma: no cover
    return None


async def run_pipeline(
    client: httpx.AsyncClient,
    payload: PipelineInput,
    *,
    on_stage: StageCallback = _noop_stage,
) -> PipelineOutput:
    validate_input(payload)

    warnings: List[str] = []
    timings: dict = {}
    pipeline_started = time.perf_counter()

    # ── Stage 1: OCR ─────────────────────────────────────────────────────────
    await on_stage("ocr", "Reading the answer sheet and marking scheme.")
    stage_started = time.perf_counter()

    # Rename uploads so two concurrent runs cannot overwrite each other inside
    # ocr-service's data/raw, which keys files by the uploaded filename stem.
    answer_images = [
        (f"{payload.run_id}_page{index}{_suffix(name)}", content, content_type)
        for index, (name, content, content_type) in enumerate(payload.answer_images, start=1)
    ]
    scheme_name, scheme_bytes, scheme_type = payload.marking_scheme_image
    scheme_image = (f"{payload.run_id}_scheme{_suffix(scheme_name)}", scheme_bytes, scheme_type)

    # The two OCR calls are independent, so overlap them.
    answer_response, scheme_response = await asyncio.gather(
        ocr_client.extract_pages(
            client,
            images=answer_images,
            question_text=payload.question_text,
            ocr_mode=payload.ocr_mode,
            use_math_ocr=payload.use_math_ocr,
        ),
        ocr_client.extract_marking_scheme(
            client, image=scheme_image, question_text=payload.question_text
        ),
    )

    payload.release_images()

    questions, ocr_warnings = normalize_ocr_questions(
        answer_response, fallback_question_text=payload.question_text
    )
    warnings.extend(ocr_warnings)

    scheme, scheme_warnings = normalize_marking_scheme(scheme_response)
    warnings.extend(scheme_warnings)

    selected, selection_warnings = select_questions(
        questions,
        policy=payload.multi_question_policy or settings.multi_question_policy,
        question_id=payload.question_id,
    )
    warnings.extend(selection_warnings)

    timings["ocr"] = int((time.perf_counter() - stage_started) * 1000)

    # ── Stages 2 & 3, per selected question ──────────────────────────────────
    question_results: List[QuestionResult] = []
    reasoning_ms = 0
    feedback_ms = 0

    for position, question in enumerate(selected, start=1):
        suffix = f" (question {position} of {len(selected)})" if len(selected) > 1 else ""

        await on_stage("reasoning", f"Checking each step against the marking scheme{suffix}.")
        stage_started = time.perf_counter()
        evaluation = await reasoning_client.evaluate(
            client, request=build_evaluation_request(question, scheme)
        )
        reasoning_ms += int((time.perf_counter() - stage_started) * 1000)

        fallback_warning = detect_reasoning_fallback(evaluation)
        if fallback_warning:
            warnings.append(f"{question.question_id}: {fallback_warning}")

        feedback_request, adapter_warnings = build_feedback_request(
            question,
            scheme,
            evaluation,
            include_unanalyzed_steps=settings.include_unanalyzed_steps,
        )
        warnings.extend(f"{question.question_id}: {w}" for w in adapter_warnings)

        await on_stage("feedback", f"{FEEDBACK_STAGE_MESSAGE}{suffix}")
        stage_started = time.perf_counter()
        feedback = await feedback_client.generate_feedback(client, request=feedback_request)
        feedback_ms += int((time.perf_counter() - stage_started) * 1000)

        question_results.append(
            QuestionResult(
                question_id=question.question_id,
                question_text=question.question_text,
                marking_scheme=scheme,
                student_steps=question.student_steps,
                final_answer=question.final_answer,
                reasoning=evaluation,
                feedback=feedback,
            )
        )

    timings["reasoning"] = reasoning_ms
    timings["feedback"] = feedback_ms
    timings["total"] = int((time.perf_counter() - pipeline_started) * 1000)

    primary = question_results[0]
    result = EvaluationResult(
        final_score=primary.feedback.final_score,
        total_marks=primary.feedback.total_marks or scheme.total_marks,
        step_feedback=primary.feedback.step_feedback,
        overall_feedback=primary.feedback.overall_feedback,
        improvement_suggestions=primary.feedback.improvement_suggestions,
        question_id=primary.question_id,
        question_text=primary.question_text,
        question_count=len(questions),
        questions=question_results,
        warnings=warnings,
        timings_ms=timings,
    )

    await on_stage("done", None)
    logger.info(
        "[pipeline] complete — %s, %.1f/%.1f marks, %dms total",
        primary.question_id,
        result.final_score,
        result.total_marks,
        timings["total"],
    )
    return PipelineOutput(result=result, warnings=warnings)


def _suffix(filename: str) -> str:
    """Keep an image extension ocr-service accepts, defaulting to .jpg."""
    lowered = (filename or "").lower()
    for extension in (".jpg", ".jpeg", ".png"):
        if lowered.endswith(extension):
            return extension
    return ".jpg"
