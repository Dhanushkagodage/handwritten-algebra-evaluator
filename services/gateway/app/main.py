"""Gateway — the integration layer for Modules 01-03.

    uvicorn app.main:app --host 127.0.0.1 --port 8080 --workers 1

Chains ocr-service -> reasoning-service -> feedback-service over HTTP and owns
all schema adaptation between them. It imports nothing from those services and
holds no credentials of its own; each module keeps its own venv, its own secrets,
and its own standalone terminal workflows.

Run with a SINGLE worker: the job store lives in this process.
"""
import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import evaluate, health, jobs
from app.clients.base import build_http_client
from app.config import settings
from app.core.handlers import register_exception_handlers
from app.services.jobs import InMemoryJobStore

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

SWEEP_INTERVAL_SECONDS = 60


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.http = build_http_client()
    app.state.jobs = InMemoryJobStore()

    async def sweeper() -> None:
        while True:
            await asyncio.sleep(SWEEP_INTERVAL_SECONDS)
            try:
                await app.state.jobs.sweep()
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001 — the sweeper must never die
                logger.exception("[jobs] sweep failed")

    app.state.sweeper = asyncio.create_task(sweeper(), name="job-sweeper")

    logger.info("Gateway ready on port %s", settings.gateway_port)
    logger.info("  ocr       -> %s", settings.ocr_service_url)
    logger.info("  reasoning -> %s", settings.reasoning_service_url)
    logger.info("  feedback  -> %s", settings.feedback_service_url)

    try:
        yield
    finally:
        app.state.sweeper.cancel()
        await asyncio.gather(app.state.sweeper, return_exceptions=True)
        await app.state.jobs.shutdown()
        await app.state.http.aclose()


app = FastAPI(
    title="Algebra Evaluation Gateway",
    version="1.0.0",
    description=(
        "One call runs OCR (Module 01), multi-agent reasoning and marking (Module 02), "
        "and stepwise feedback generation (Module 03), returning a graded, explained "
        "result. The three services are untouched and remain independently runnable."
    ),
    lifespan=lifespan,
)

# The browser only ever talks to the gateway, so this is the single place CORS
# needs configuring — the per-service CORS settings stop mattering.
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.gateway_cors_origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

register_exception_handlers(app)

app.include_router(health.router)
app.include_router(evaluate.router)
app.include_router(jobs.router)
