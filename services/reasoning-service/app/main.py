"""
FastAPI application entry point.

Startup order (critical for Windows multiprocessing compatibility):
1. Load .env FIRST before any langchain/openai imports
2. Configure logging
3. Import and mount routers
4. Register lifespan events

Run with:
    uvicorn app.main:app --host 0.0.0.0 --port 8002 --reload
"""
import logging
import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv

# ── MUST load env before any LangChain / OpenAI imports ──────────────────────
load_dotenv()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# ── Configure structured logging ──────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: startup logs and teardown."""
    logger.info("=" * 60)
    logger.info("Algebra Evaluation Service starting up")
    logger.info("LLM provider : %s", os.getenv("LLM_PROVIDER", "openai"))
    logger.info("LLM model    : %s", os.getenv("LLM_MODEL", "gpt-4o-mini"))
    logger.info("Port         : %s", os.getenv("PORT", "8002"))
    logger.info("=" * 60)
    yield
    logger.info("Algebra Evaluation Service shut down")


# ── Create FastAPI app ────────────────────────────────────────────────────────
app = FastAPI(
    title="Handwritten Algebra Evaluator — Reasoning Service",
    description=(
        "Multi-agent LangGraph service for evaluating A/L algebra student answers. "
        "Runs Step Validation, Method Detection, and Scheme Matching agents in parallel, "
        "then synthesises results via a Supervisor Agent."
    ),
    version="2.0.0",
    lifespan=lifespan,
)

# ── CORS ──────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Mount routers ─────────────────────────────────────────────────────────────
from app.api.routes import router as evaluation_router  # noqa: E402

app.include_router(evaluation_router)


@app.get("/health", tags=["Health"])
async def health_check():
    """Simple liveness probe."""
    return {"status": "ok", "service": "reasoning-service", "version": "2.0.0"}