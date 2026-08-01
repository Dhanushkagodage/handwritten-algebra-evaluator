"""End-to-end pipeline tests against a mocked transport.

No services, no credentials, no network — httpx.MockTransport stands in for all
three modules, so the full chain including the job store is exercised offline.
"""
import asyncio

import httpx
import pytest

from app.core.errors import GatewayError, InvalidInputError, JobNotFoundError
from app.services.jobs import InMemoryJobStore
from app.services.pipeline import PipelineInput, run_pipeline

OCR_ANSWER = {
    "reasoning_input": {
        "question_text": "Solve x^2 - 5x + 6 = 0",
        "student_steps": [
            {"step_id": 1, "content": "x^2 - 5x + 6 = 0"},
            {"step_id": 2, "content": "(x - 2)(x - 3) = 0"},
            {"step_id": 3, "content": "x = 2 or x = 3"},
        ],
        "final_answer": "x = 2, x = 3",
    }
}

OCR_SCHEME = {
    "marking_scheme": {
        "total_marks": 6.0,
        "steps": [
            {"step_no": 1, "description": "Write in standard form", "expected_expression": "x^2 - 5x + 6 = 0", "marks": 2.0},
            {"step_no": 2, "description": "Factorise", "expected_expression": "(x - 2)(x - 3) = 0", "marks": 2.0},
            {"step_no": 3, "description": "State the roots", "expected_expression": "x = 2, x = 3", "marks": 2.0},
        ],
    }
}

REASONING_OUTPUT = {
    "steps_analysis": [
        {"step_id": 1, "validity": True, "status": "correct", "method": "factorisation",
         "matched_scheme_step": 1, "match_score": 1.0, "marks_awarded": 2.0, "max_marks": 2.0, "confidence": 0.95},
        {"step_id": 2, "validity": True, "status": "partially_correct", "method": "factorisation",
         "matched_scheme_step": 2, "match_score": 0.7, "marks_awarded": 1.0, "max_marks": 2.0, "confidence": 0.8},
        {"step_id": 3, "validity": False, "status": "unclear", "method": "factorisation",
         "matched_scheme_step": 3, "match_score": 0.3, "marks_awarded": 0.0, "max_marks": 2.0, "confidence": 0.5},
    ],
    "total_marks": 3.0,
    "max_marks": 6.0,
    "percentage": 50.0,
    "summary": "Correct method, incomplete final statement.",
    "method_feedback": "Factorisation was appropriate.",
    "step_validation": {
        "step_validations": [
            {"step_id": 2, "is_valid": True, "status": "partially_correct", "error": "Factors not fully simplified", "confidence": 0.8}
        ],
        "missing_transitions": [],
    },
    "method_detection": {"detected_method": "factorisation", "method_is_valid": True,
                         "alternative_methods_possible": True, "alternative_methods": ["quadratic formula"], "confidence": 0.9},
}

FEEDBACK_OUTPUT = {
    "final_score": 3.0,
    "total_marks": 6.0,
    "step_feedback": [
        {"step_number": 1, "expression": "x^2 - 5x + 6 = 0", "validity": "correct", "marks_awarded": 2.0,
         "what_is_correct": "Standard form is right.", "how_to_improve": "Keep doing this.", "feedback": "Good start."},
        {"step_number": 2, "expression": "(x - 2)(x - 3) = 0", "validity": "partial", "marks_awarded": 1.0,
         "what_is_correct": "Factors are right.", "what_is_missing": "Simplification.",
         "why_marks_reduced": "Incomplete working.", "how_to_improve": "Show the full factorisation.",
         "feedback": "Nearly there."},
        {"step_number": 3, "expression": "x = 2 or x = 3", "validity": "incorrect", "marks_awarded": 0.0,
         "what_is_correct": "You attempted the roots.", "what_is_missing": "A clear final answer.",
         "why_marks_reduced": "The statement was unclear.", "how_to_improve": "Write x = 2, x = 3.",
         "feedback": "State the roots clearly."},
    ],
    "overall_feedback": "Solid method, finish the working.",
    "improvement_suggestions": ["Always state both roots explicitly."],
}


def make_client(handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler), trust_env=False)


def happy_path(seen=None):
    def handler(request: httpx.Request) -> httpx.Response:
        if seen is not None:
            seen.append(str(request.url))
        path = request.url.path
        if path == "/extract-pages":
            return httpx.Response(200, json=OCR_ANSWER)
        if path == "/extract-marking-scheme":
            return httpx.Response(200, json=OCR_SCHEME)
        if path == "/api/v1/evaluate":
            return httpx.Response(200, json=REASONING_OUTPUT)
        if path == "/api/v1/feedback":
            return httpx.Response(200, json=FEEDBACK_OUTPUT)
        return httpx.Response(404, json={"detail": f"unexpected path {path}"})

    return handler


def sample_input(**overrides) -> PipelineInput:
    payload = dict(
        answer_images=[("answer.jpg", b"fake-jpeg-bytes", "image/jpeg")],
        marking_scheme_image=("scheme.jpg", b"fake-jpeg-bytes", "image/jpeg"),
        question_text="Solve x^2 - 5x + 6 = 0",
        run_id="testrun",
    )
    payload.update(overrides)
    return PipelineInput(**payload)


# ── Happy path ───────────────────────────────────────────────────────────────

async def test_full_pipeline_produces_a_flattened_result():
    async with make_client(happy_path()) as client:
        output = await run_pipeline(client, sample_input())

    result = output.result
    assert result.final_score == 3.0
    assert result.total_marks == 6.0
    assert len(result.step_feedback) == 3
    assert result.overall_feedback == "Solid method, finish the working."
    assert result.improvement_suggestions == ["Always state both roots explicitly."]
    assert result.question_id == "Q1"
    assert result.question_count == 1
    assert set(result.timings_ms) == {"ocr", "reasoning", "feedback", "total"}


async def test_result_carries_the_full_per_question_detail():
    async with make_client(happy_path()) as client:
        output = await run_pipeline(client, sample_input())

    question = output.result.questions[0]
    assert question.question_text == "Solve x^2 - 5x + 6 = 0"
    assert question.marking_scheme.total_marks == 6.0
    assert [s.step_id for s in question.student_steps] == [1, 2, 3]
    # Module 02's own output is preserved for the results page.
    assert question.reasoning.percentage == 50.0
    assert question.reasoning.method_detection.detected_method == "factorisation"


async def test_uploads_are_renamed_so_concurrent_runs_cannot_collide():
    """ocr-service stores raw uploads under the uploaded filename stem, so two
    runs both sending answer.jpg would overwrite each other."""
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/extract-pages":
            captured["answer"] = request.content
        if request.url.path == "/extract-marking-scheme":
            captured["scheme"] = request.content
        return happy_path()(request)

    async with make_client(handler) as client:
        await run_pipeline(client, sample_input())

    assert b"testrun_page1.jpg" in captured["answer"]
    assert b"answer.jpg" not in captured["answer"]
    assert b"testrun_scheme.jpg" in captured["scheme"]


async def test_image_bytes_are_released_after_ocr():
    payload = sample_input()
    async with make_client(happy_path()) as client:
        await run_pipeline(client, payload)
    assert payload.answer_images == []


async def test_stage_callbacks_fire_in_order():
    stages = []

    async def on_stage(stage, message=None):
        stages.append(stage)

    async with make_client(happy_path()) as client:
        await run_pipeline(client, sample_input(), on_stage=on_stage)

    assert stages == ["ocr", "reasoning", "feedback", "done"]


async def test_the_adapter_translates_reasoning_output_for_feedback():
    """The one hop where the contracts genuinely disagree."""
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/v1/feedback":
            import json

            captured.update(json.loads(request.content))
        return happy_path()(request)

    async with make_client(handler) as client:
        await run_pipeline(client, sample_input())

    steps = captured["student_steps"]
    assert [s["step_number"] for s in steps] == [1, 2, 3]
    # status -> validity, including the "unclear" resolution
    assert [s["validity"] for s in steps] == ["correct", "partial", "incorrect"]
    # expression is joined in from the OCR output; reasoning never supplies it
    assert steps[0]["expression"] == "x^2 - 5x + 6 = 0"
    assert steps[1]["error_description"] == "Factors not fully simplified"
    assert captured["detected_method"] == "factorisation"
    assert captured["assigned_marks"] == 3.0
    # neither of these is echoed back by reasoning — the gateway carries them
    assert captured["question_text"] == "Solve x^2 - 5x + 6 = 0"
    assert captured["marking_scheme"]["total_marks"] == 6.0


# ── Input validation ─────────────────────────────────────────────────────────

async def test_no_answer_images_is_rejected():
    async with make_client(happy_path()) as client:
        with pytest.raises(InvalidInputError):
            await run_pipeline(client, sample_input(answer_images=[]))


async def test_too_many_answer_images_is_rejected():
    images = [(f"p{n}.jpg", b"x", "image/jpeg") for n in range(9)]
    async with make_client(happy_path()) as client:
        with pytest.raises(InvalidInputError):
            await run_pipeline(client, sample_input(answer_images=images))


async def test_bad_ocr_mode_is_rejected_before_any_call():
    async with make_client(happy_path()) as client:
        with pytest.raises(InvalidInputError):
            await run_pipeline(client, sample_input(ocr_mode="magic"))


# ── Failure propagation ──────────────────────────────────────────────────────

async def test_unreachable_service_becomes_a_503_naming_the_service_and_url():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    async with make_client(handler) as client:
        with pytest.raises(GatewayError) as excinfo:
            await run_pipeline(client, sample_input())

    error = excinfo.value
    assert error.status_code == 503
    assert error.stage == "ocr"
    assert "OCR service (Module 01)" in error.message
    assert "Is it running?" in error.message


async def test_upstream_error_is_attributed_to_its_stage():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/v1/feedback":
            return httpx.Response(500, json={"detail": "Space did not wake up in time"})
        return happy_path()(request)

    async with make_client(handler) as client:
        with pytest.raises(GatewayError) as excinfo:
            await run_pipeline(client, sample_input())

    error = excinfo.value
    assert error.status_code == 502
    assert error.error_code == "FEEDBACK_FAILED"
    assert error.stage == "feedback"
    assert error.message == "Space did not wake up in time"


async def test_reasoning_error_envelope_is_surfaced_verbatim():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/v1/evaluate":
            return httpx.Response(
                400,
                json={"error": True, "error_code": "EMPTY_SCHEME",
                      "message": "Marking scheme has no steps.", "details": {}, "status_code": 400},
            )
        return happy_path()(request)

    async with make_client(handler) as client:
        with pytest.raises(GatewayError) as excinfo:
            await run_pipeline(client, sample_input())

    error = excinfo.value
    assert error.stage == "reasoning"
    assert error.message == "Marking scheme has no steps."
    assert error.details["upstream_error_code"] == "EMPTY_SCHEME"


async def test_zero_total_marks_never_reaches_the_result_as_a_nan_divide():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/extract-marking-scheme":
            scheme = {"marking_scheme": {**OCR_SCHEME["marking_scheme"], "total_marks": 0.0}}
            return httpx.Response(200, json=scheme)
        return happy_path()(request)

    async with make_client(handler) as client:
        output = await run_pipeline(client, sample_input())

    assert output.result.total_marks > 0
    assert any("showed no total" in w for w in output.warnings)


# ── Job store ────────────────────────────────────────────────────────────────

async def wait_for(store: InMemoryJobStore, job_id: str, *, timeout: float = 5.0):
    deadline = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < deadline:
        job = await store.get(job_id)
        if job.status in {"succeeded", "failed", "cancelled"}:
            return job
        await asyncio.sleep(0.01)
    raise AssertionError(f"job {job_id} did not finish within {timeout}s")


async def test_job_runs_to_completion_and_records_each_stage():
    store = InMemoryJobStore()
    async with make_client(happy_path()) as client:
        job = await store.create(client, sample_input())
        assert job.status in {"pending", "running"}
        finished = await wait_for(store, job.job_id)

    assert finished.status == "succeeded"
    assert finished.stage == "done"
    assert finished.result.final_score == 3.0

    response = finished.to_response()
    assert [s.key for s in response.stages] == ["ocr", "reasoning", "feedback"]
    assert all(s.status == "succeeded" for s in response.stages)
    assert all(s.duration_ms is not None for s in response.stages)
    assert response.progress == 1.0
    assert response.elapsed_ms >= 0


async def test_failed_job_reports_the_failing_stage_rather_than_raising():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/v1/feedback":
            return httpx.Response(503, json={"detail": "Space unavailable"})
        return happy_path()(request)

    store = InMemoryJobStore()
    async with make_client(handler) as client:
        job = await store.create(client, sample_input())
        finished = await wait_for(store, job.job_id)

    assert finished.status == "failed"
    assert finished.error.stage == "feedback"
    assert finished.error.error_code == "FEEDBACK_FAILED"
    # `detail` mirrors `message` so existing frontend error handling still works.
    assert finished.error.detail == finished.error.message

    response = finished.to_response()
    failed = next(s for s in response.stages if s.key == "feedback")
    assert failed.status == "failed"


async def test_job_payload_is_freed_when_the_job_finishes():
    store = InMemoryJobStore()
    async with make_client(happy_path()) as client:
        job = await store.create(client, sample_input())
        finished = await wait_for(store, job.job_id)
    assert finished.payload is None


async def test_unknown_job_id_is_a_404():
    store = InMemoryJobStore()
    with pytest.raises(JobNotFoundError):
        await store.get("does-not-exist")


async def test_cancelling_a_job_marks_it_cancelled():
    started = asyncio.Event()

    def handler(request: httpx.Request) -> httpx.Response:
        started.set()
        return happy_path()(request)

    store = InMemoryJobStore()
    async with make_client(handler) as client:
        job = await store.create(client, sample_input())
        await started.wait()
        await store.cancel(job.job_id)
        await asyncio.gather(job.task, return_exceptions=True)

    assert (await store.get(job.job_id)).status in {"cancelled", "succeeded"}
