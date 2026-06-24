from fastapi import APIRouter, HTTPException, Request

from app.models.schemas import FeedbackRequest, FeedbackResponse

router = APIRouter()


@router.post("/feedback", response_model=FeedbackResponse)
async def generate_feedback(request_body: FeedbackRequest, request: Request):
    try:
        generator = request.app.state.generator
        result = await generator.generate(request_body)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
