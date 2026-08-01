"""POST/GET/DELETE /api/v1/jobs — the async pipeline the frontend drives."""
import uuid
from typing import List, Optional

from fastapi import APIRouter, Request, Response, UploadFile, status

from app.api import uploads
from app.schemas.gateway import JobCreatedResponse, JobStatusResponse

router = APIRouter(prefix="/api/v1/jobs", tags=["Jobs"])


@router.post(
    "",
    response_model=JobCreatedResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Start an evaluation and return immediately",
    description=(
        "Uploads are read up front, then the pipeline runs in the background. Poll the "
        "returned `poll_url` to watch `stage` advance through ocr -> reasoning -> "
        "feedback -> done."
    ),
)
async def create_job(
    request: Request,
    answer_images: List[UploadFile] = uploads.AnswerImages,
    marking_scheme_image: UploadFile = uploads.SchemeImage,
    question_text: str = uploads.QuestionText,
    ocr_mode: Optional[str] = uploads.OcrMode,
    use_math_ocr: Optional[bool] = uploads.UseMathOcr,
    question_id: Optional[str] = uploads.QuestionId,
    multi_question_policy: Optional[str] = uploads.MultiQuestionPolicy,
) -> JobCreatedResponse:
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
    job = await request.app.state.jobs.create(request.app.state.http, payload)
    return JobCreatedResponse(
        job_id=job.job_id,
        status=job.status,
        stage=job.stage,
        poll_url=f"/api/v1/jobs/{job.job_id}",
    )


@router.get(
    "/{job_id}",
    response_model=JobStatusResponse,
    summary="Poll an evaluation job",
    description=(
        "Always answers 200 for a known job, so a poller has a single happy path. A "
        "failed run reports `status: \"failed\"` with `error.stage` naming the module "
        "that failed. Unknown or expired ids answer 404."
    ),
)
async def get_job(request: Request, job_id: str) -> JobStatusResponse:
    job = await request.app.state.jobs.get(job_id)
    return job.to_response()


@router.delete(
    "/{job_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Cancel a running evaluation",
)
async def cancel_job(request: Request, job_id: str) -> Response:
    await request.app.state.jobs.cancel(job_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)
