"""
FastAPI application entry point.

Startup order (critical for Windows multiprocessing compatibility):
1. Load .env FIRST before any langchain/openai imports
2. Configure logging
3. Import settings (validates all env vars at startup)
4. Create app, register handlers, mount routers
5. Register lifespan events

Run with:
    uvicorn app.main:app --host 0.0.0.0 --port 8002 --reload
"""
import logging
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

# ── Load & validate settings (fails fast if config is bad) ────────────────────
from app.config.settings import settings
from app.core.handlers import register_exception_handlers


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: startup logs and teardown."""
    logger.info("=" * 60)
    logger.info("Algebra Evaluation Service starting up")
    logger.info("LLM provider : %s", settings.llm_provider)
    logger.info("LLM model    : %s", settings.llm_model)
    logger.info("Port         : %s", settings.port)
    logger.info("=" * 60)
    yield
    logger.info("Algebra Evaluation Service shut down")


# ── Create FastAPI app ────────────────────────────────────────────────────────
app = FastAPI(
    title=settings.app_name,
    description=(
        "Multi-agent LangGraph service for evaluating A/L algebra student answers. "
        "Runs Step Validation, Method Detection, and Scheme Matching agents in parallel, "
        "then synthesises results via a Supervisor Agent."
    ),
    version=settings.app_version,
    lifespan=lifespan,
)

# ── Register global exception handlers ────────────────────────────────────────
register_exception_handlers(app)

# ── CORS ──────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Mount routers ─────────────────────────────────────────────────────────────
from app.api.routes import router as evaluation_router  # noqa: E402
from app.single_llm.routes import router as single_llm_router  # noqa: E402

app.include_router(evaluation_router)
app.include_router(single_llm_router)


@app.get("/health", tags=["Health"])
async def health_check():
    """Simple liveness probe."""
    return {
        "status": "ok",
        "service": "reasoning-service",
        "version": settings.app_version,
        "llm_provider": settings.llm_provider,
        "llm_model": settings.llm_model,
    }