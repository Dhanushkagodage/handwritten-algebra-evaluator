from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.agents.feedback_generator import FeedbackGenerator
from app.routers import feedback

load_dotenv()


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.generator = FeedbackGenerator()
    await app.state.generator.load_model()
    yield
    app.state.generator = None


app = FastAPI(
    title="Feedback Service",
    version="1.0.0",
    description="Stepwise feedback generation using Gemma 3n + LoRA",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(feedback.router, prefix="/api/v1", tags=["feedback"])


@app.get("/health")
def health_check():
    return {"status": "ok", "service": "feedback-service"}
