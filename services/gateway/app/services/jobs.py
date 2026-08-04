"""In-process job store.

Deliberately not Celery/Redis: concurrency here is one demo at a time, and a
broker would add a Windows-hostile dependency for benefits this project cannot
use. The trade-off is that the gateway MUST run with a single worker — with two,
a job created on worker A 404s on worker B — and `--reload` wipes the store.
"""
import asyncio
import logging
import uuid
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

import httpx

from app.config import settings
from app.core.errors import GatewayError, JobNotFoundError
from app.core.handlers import envelope_from_gateway_error
from app.schemas.gateway import (
    ErrorEnvelope,
    EvaluationResult,
    JobStatus,
    JobStatusResponse,
    StageKey,
    StageRecord,
)
from app.services.pipeline import PipelineInput, run_pipeline

logger = logging.getLogger(__name__)

STAGE_ORDER: List[StageKey] = ["ocr", "reasoning", "feedback"]


def _now() -> datetime:
    return datetime.now(timezone.utc)


class Job:
    def __init__(self, job_id: str, payload: PipelineInput) -> None:
        self.job_id = job_id
        self.payload: Optional[PipelineInput] = payload
        self.status: JobStatus = "pending"
        self.stage: StageKey = "queued"
        self.stage_message: Optional[str] = None
        self.stages: Dict[str, StageRecord] = {
            key: StageRecord(key=key) for key in STAGE_ORDER
        }
        self.warnings: List[str] = []
        self.result: Optional[EvaluationResult] = None
        self.error: Optional[ErrorEnvelope] = None
        self.created_at = _now()
        self.updated_at = self.created_at
        self.finished_at: Optional[datetime] = None
        # Hold a strong reference: a bare asyncio.create_task() result can be
        # garbage-collected mid-flight.
        self.task: Optional[asyncio.Task] = None

    @property
    def progress(self) -> float:
        if self.status == "succeeded":
            return 1.0
        done = sum(1 for record in self.stages.values() if record.status == "succeeded")
        return round(done / len(STAGE_ORDER), 2)

    def to_response(self) -> JobStatusResponse:
        end = self.finished_at or _now()
        return JobStatusResponse(
            job_id=self.job_id,
            status=self.status,
            stage=self.stage,
            stage_message=self.stage_message,
            stages=[self.stages[key] for key in STAGE_ORDER],
            progress=self.progress,
            created_at=self.created_at,
            updated_at=self.updated_at,
            elapsed_ms=int((end - self.created_at).total_seconds() * 1000),
            poll_after_ms=1500,
            warnings=self.warnings,
            result=self.result,
            error=self.error,
        )


class InMemoryJobStore:
    def __init__(self) -> None:
        self._jobs: Dict[str, Job] = {}
        self._lock = asyncio.Lock()
        self._semaphore = asyncio.Semaphore(settings.job_max_concurrency)

    async def create(self, client: httpx.AsyncClient, payload: PipelineInput) -> Job:
        job = Job(uuid.uuid4().hex, payload)
        async with self._lock:
            await self._evict_if_needed()
            self._jobs[job.job_id] = job
        job.task = asyncio.create_task(self._run(client, job), name=f"job-{job.job_id}")
        return job

    async def get(self, job_id: str) -> Job:
        async with self._lock:
            job = self._jobs.get(job_id)
        if job is None:
            raise JobNotFoundError(
                f"No evaluation job with id '{job_id}'. It may have expired — "
                f"jobs are kept for {settings.job_ttl_seconds // 60} minutes."
            )
        return job

    async def cancel(self, job_id: str) -> None:
        job = await self.get(job_id)
        if job.task and not job.task.done():
            job.task.cancel()

    async def _run(self, client: httpx.AsyncClient, job: Job) -> None:
        async with self._semaphore:
            payload = job.payload
            if payload is None:  # pragma: no cover — defensive
                return
            try:
                job.status = "running"
                job.updated_at = _now()

                async def on_stage(stage: str, message: Optional[str] = None) -> None:
                    self._advance(job, stage, message)

                output = await run_pipeline(client, payload, on_stage=on_stage)

                job.result = output.result
                job.warnings = output.warnings
                job.status = "succeeded"
                job.stage = "done"
                job.stage_message = None

            except asyncio.CancelledError:
                self._fail_current_stage(job, "cancelled")
                job.status = "cancelled"
                job.stage_message = "The evaluation was cancelled."
                raise

            except GatewayError as exc:
                self._fail_current_stage(job, "failed")
                job.status = "failed"
                job.error = envelope_from_gateway_error(exc)
                job.stage_message = exc.message
                logger.warning("[job %s] failed at %s: %s", job.job_id, exc.stage, exc.message)

            except Exception as exc:  # noqa: BLE001 — a job must never crash silently
                self._fail_current_stage(job, "failed")
                job.status = "failed"
                message = "The gateway hit an unexpected error. Check the gateway logs."
                job.error = ErrorEnvelope(
                    error_code="INTERNAL_ERROR",
                    message=message,
                    detail=message,
                    stage=job.stage if job.stage in STAGE_ORDER else None,
                    status_code=500,
                )
                job.stage_message = message
                logger.exception("[job %s] unexpected failure: %s", job.job_id, exc)

            finally:
                job.payload = None
                job.finished_at = _now()
                job.updated_at = job.finished_at

    def _advance(self, job: Job, stage: str, message: Optional[str]) -> None:
        now = _now()

        # Close whichever stage was running.
        for record in job.stages.values():
            if record.status == "running":
                record.status = "succeeded"
                record.finished_at = now
                if record.started_at:
                    record.duration_ms = int(
                        (now - record.started_at).total_seconds() * 1000
                    )

        if stage in job.stages:
            record = job.stages[stage]
            # A second question re-enters reasoning/feedback; keep the first start.
            if record.started_at is None:
                record.started_at = now
            record.status = "running"
            record.message = message
            job.stage = stage  # type: ignore[assignment]
        elif stage == "done":
            job.stage = "done"

        job.stage_message = message
        job.updated_at = now

    def _fail_current_stage(self, job: Job, status: str) -> None:
        now = _now()
        for record in job.stages.values():
            if record.status == "running":
                record.status = "failed" if status == "failed" else "pending"
                record.finished_at = now
                if record.started_at:
                    record.duration_ms = int(
                        (now - record.started_at).total_seconds() * 1000
                    )

    async def _evict_if_needed(self) -> None:
        if len(self._jobs) < settings.max_jobs:
            return
        finished = sorted(
            (job for job in self._jobs.values() if job.finished_at is not None),
            key=lambda job: job.finished_at,  # type: ignore[arg-type]
        )
        for job in finished[: max(1, len(self._jobs) - settings.max_jobs + 1)]:
            self._jobs.pop(job.job_id, None)

    async def sweep(self) -> int:
        """Drop jobs whose TTL has expired. Returns how many were removed."""
        cutoff = _now() - timedelta(seconds=settings.job_ttl_seconds)
        async with self._lock:
            expired = [
                job_id
                for job_id, job in self._jobs.items()
                if job.finished_at is not None and job.finished_at < cutoff
            ]
            for job_id in expired:
                self._jobs.pop(job_id, None)
        if expired:
            logger.info("[jobs] swept %d expired job(s)", len(expired))
        return len(expired)

    async def shutdown(self) -> None:
        async with self._lock:
            tasks = [job.task for job in self._jobs.values() if job.task and not job.task.done()]
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
