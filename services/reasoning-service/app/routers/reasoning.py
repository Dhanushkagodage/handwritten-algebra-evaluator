from fastapi import APIRouter
from app.models.schemas import StepRequest
from app.graph.workflow import build_graph

router = APIRouter()

graph = build_graph()

@router.post("/evaluate-steps")
def evaluate_steps(request: StepRequest):
       
    result = graph.invoke({
        "question": request.question,
        "student_answer": [s.model_dump() for s in request.student_answer]
    })

    return result