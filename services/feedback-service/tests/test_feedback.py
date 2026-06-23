"""
Standalone test for Module 03 — Stepwise Feedback Generation.

Loads a sample FeedbackRequest (the kind of payload Module 02 ultimately feeds in),
runs the local Qwen2.5 SLM generator directly (no FastAPI server needed), and prints +
validates the generated stepwise feedback.

Run from the feedback-service directory:
    python tests\\test_feedback.py

The first run downloads the base model (Qwen/Qwen2.5-1.5B-Instruct by default); later
runs are offline. Override the model via the BASE_MODEL env var / .env.
"""
import asyncio
import json
import sys
from pathlib import Path

# Make `app` importable when run as a plain script from the service root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

load_dotenv()

from app.agents.feedback_generator import FeedbackGenerator
from app.models.schemas import FeedbackRequest, StepValidity

SAMPLE_PATH = Path(__file__).resolve().parent / "sample_input.json"


async def run() -> None:
    with open(SAMPLE_PATH, "r", encoding="utf-8") as f:
        request = FeedbackRequest.model_validate(json.load(f))

    print(f"Question: {request.question_text}")
    print(f"Method  : {request.detected_method}")
    print(f"Score   : {request.assigned_marks} / {request.total_marks}")
    print("Loading model (first run downloads weights)...\n")

    generator = FeedbackGenerator()
    await generator.load_model()
    response = await generator.generate(request)

    print("=" * 70)
    print("GENERATED FEEDBACK")
    print("=" * 70)
    print(json.dumps(response.model_dump(), indent=2, ensure_ascii=False))

    # ── Spec invariants ──────────────────────────────────────────────────────
    assert len(response.step_feedback) == len(request.student_steps), (
        "Expected one feedback block per student step"
    )
    for sf in response.step_feedback:
        if sf.validity != StepValidity.CORRECT:
            assert sf.why_marks_reduced, (
                f"Step {sf.step_number} lost marks but has no deduction explanation"
            )
    assert response.final_score == request.assigned_marks
    assert response.total_marks == request.total_marks
    assert response.improvement_suggestions, "Expected at least one improvement suggestion"

    print("\nAll assertions passed.")


if __name__ == "__main__":
    asyncio.run(run())
