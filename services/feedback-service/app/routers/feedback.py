from fastapi import APIRouter, HTTPException

from app.agents.feedback_generator import FeedbackGenerator
from app.models.schemas import FeedbackRequest, FeedbackResponse

router = APIRouter()
generator = FeedbackGenerator()


@router.post("/feedback", response_model=FeedbackResponse)
async def generate_feedback(request: FeedbackRequest):
    try:
        result = await generator.generate(request)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
