import asyncio
import json
import os
import re
from typing import Dict, List, Optional

from gradio_client import Client
from gradio_client.exceptions import AppError

from app.models.schemas import (
    FeedbackRequest,
    FeedbackResponse,
    StepFeedback,
    StepResult,
    StepValidity,
)
from app.shared_format import FORMAT_INSTRUCTION as _FORMAT_INSTRUCTION


class FeedbackGenerator:
    """
    Module 03 — Stepwise Feedback Generation.

    Calls your own LoRA-fine-tuned Qwen2.5-3B-Instruct — trained on Colab
    using the algebra dataset in app/training/dataset.py — over Gradio's
    API client. The model is served by a persistent Hugging Face Space
    (app/serving/hf_space/app.py, Gradio SDK + ZeroGPU) that pulls the
    adapter from Hugging Face Hub, reached via SPACE_ID + API_KEY. See
    app/serving/hf_space/DEPLOY.md for how to set it up.

    Generates four-component per-step feedback:
      1. What is correct
      2. What is missing / incorrect
      3. Why marks were reduced
      4. How to improve
    """

    def __init__(self):
        self._loaded = False
        self._space_id: Optional[str] = None
        self._api_key: Optional[str] = None
        self._client: Optional[Client] = None

    async def load_model(self) -> None:
        if self._loaded:
            return
        self._load_space_client()
        self._loaded = True

    def _load_space_client(self) -> None:
        self._space_id = os.getenv("SPACE_ID")
        self._api_key = os.getenv("API_KEY")
        print(f"[FeedbackGenerator] calling trained model at {self._space_id}")

    def _get_client(self) -> Client:
        # Client(...) does a handshake to fetch the Space's API schema, so
        # build it once and reuse it — not on every request.
        if self._client is None:
            self._client = Client(self._space_id)
        return self._client

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    async def generate(self, request: FeedbackRequest) -> FeedbackResponse:
        if not self._loaded:
            await self.load_model()

        messages = self._build_messages(request)
        raw_text = await self._run_inference(messages)

        step_feedback = self._parse_step_feedback(raw_text, request.student_steps)
        step_feedback = self._validate_feedback(step_feedback)

        return FeedbackResponse(
            final_score=request.assigned_marks,
            total_marks=request.total_marks,
            step_feedback=step_feedback,
            overall_feedback=self._build_overall_feedback(request, step_feedback),
            improvement_suggestions=self._extract_suggestions(step_feedback),
        )

    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------

    def _build_messages(self, request: FeedbackRequest) -> List[Dict]:
        scheme_marks = {m.step_number: m.marks for m in request.marking_scheme}

        steps_text = "\n".join(
            "Step {n}: {expr} [{v}, {awarded}/{possible} marks]{err}".format(
                n=s.step_number,
                expr=s.expression,
                v=s.validity.value.upper(),
                awarded=s.marks_awarded,
                possible=scheme_marks.get(s.step_number, "?"),
                err=f" — Error: {s.error_description}" if s.error_description else "",
            )
            for s in request.student_steps
        )

        scheme_text = "\n".join(
            "Step {n}: {expr} [{m} marks]{desc}".format(
                n=m.step_number,
                expr=m.expected_expression,
                m=m.marks,
                desc=f" — {m.description}" if m.description else "",
            )
            for m in request.marking_scheme
        )

        user_content = (
            f"Question: {request.question_text}\n"
            f"Solution Method: {request.detected_method}\n"
            f"Score: {request.assigned_marks} / {request.total_marks}\n\n"
            f"Student's Steps:\n{steps_text}\n\n"
            f"Marking Scheme:\n{scheme_text}\n"
            f"{_FORMAT_INSTRUCTION}"
        )

        return [
            {
                "role": "system",
                "content": "You are an algebra teacher giving feedback on a student's exam answer.",
            },
            {"role": "user", "content": user_content},
        ]

    # ------------------------------------------------------------------
    # Inference — your fine-tuned model, called over the HF Space
    # ------------------------------------------------------------------

    async def _run_inference(self, messages: List[Dict]) -> str:
        return await self._run_space_inference(messages)

    # Free HF Spaces sleep when idle and take a while to wake on the next
    # request (plus ZeroGPU's own GPU-allocation wait on a cold Space); a
    # request landing during that window sees a connection failure before
    # the container/GPU is ready, not a slow response.
    _SPACE_RETRY_DELAYS = (5, 15, 30)

    async def _run_space_inference(self, messages: List[Dict]) -> str:
        messages_json = json.dumps(messages)

        last_error: Optional[Exception] = None
        for delay in (0,) + self._SPACE_RETRY_DELAYS:
            if delay:
                await asyncio.sleep(delay)
            try:
                client = await asyncio.to_thread(self._get_client)
                text = await asyncio.to_thread(
                    client.predict,
                    messages_json,
                    self._api_key,
                    api_name="/generate",
                )
                return text.strip()
            except AppError:
                # Server-side gr.Error (e.g. bad api_key) — retrying won't help.
                raise
            except Exception as exc:
                last_error = exc
        raise last_error

    # ------------------------------------------------------------------
    # Response parsing
    # ------------------------------------------------------------------

    def _parse_step_feedback(
        self, raw_text: str, steps: List[StepResult]
    ) -> List[StepFeedback]:
        result = []
        for step in steps:
            block = self._find_step_block(raw_text, step.step_number)

            correct = (self._extract_field(block, "CORRECT") if block else None) or (
                "Correct." if step.validity == StepValidity.CORRECT else "Step acknowledged."
            )
            missing = self._extract_field(block, "MISSING") if block else None
            deduction = self._extract_field(block, "DEDUCTION") if block else None
            improve = (self._extract_field(block, "IMPROVE") if block else None) or (
                "Well done, continue to the next step."
                if step.validity == StepValidity.CORRECT
                else "Review this step and practice similar problems."
            )

            parts = [f"✓ {correct}"]
            if missing:
                parts.append(f"✗ {missing}")
            if deduction:
                parts.append(f"⚠ {deduction}")
            parts.append(f"→ {improve}")

            result.append(
                StepFeedback(
                    step_number=step.step_number,
                    expression=step.expression,
                    validity=step.validity,
                    marks_awarded=step.marks_awarded,
                    what_is_correct=correct,
                    what_is_missing=missing,
                    why_marks_reduced=deduction,
                    how_to_improve=improve,
                    feedback=" ".join(parts),
                )
            )
        return result

    def _find_step_block(self, text: str, step_num: int) -> Optional[str]:
        pattern = (
            rf"===\s*STEP\s+{step_num}\s*\[.*?\]\s*===\s*\n"
            rf"(.*?)(?=\s*===\s*STEP\s+\d+|\Z)"
        )
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        return match.group(1).strip() if match else None

    def _extract_field(self, block: str, field: str) -> Optional[str]:
        pattern = rf"^{field}:\s*(.+?)(?=\n[A-Z]+:|\Z)"
        match = re.search(pattern, block, re.DOTALL | re.MULTILINE)
        return match.group(1).strip() if match else None

    # ------------------------------------------------------------------
    # Validation (per project spec)
    # ------------------------------------------------------------------

    def _validate_feedback(self, step_feedback: List[StepFeedback]) -> List[StepFeedback]:
        """Ensure every mark-lost step has a deduction explanation."""
        validated = []
        for sf in step_feedback:
            if sf.validity != StepValidity.CORRECT and not sf.why_marks_reduced:
                sf = sf.model_copy(
                    update={
                        "why_marks_reduced": (
                            f"Marks reduced because this step is {sf.validity.value} "
                            "— see the missing information above."
                        )
                    }
                )
            validated.append(sf)
        return validated

    # ------------------------------------------------------------------
    # Overall feedback and suggestions
    # ------------------------------------------------------------------

    def _build_overall_feedback(
        self, request: FeedbackRequest, step_feedback: List[StepFeedback]
    ) -> str:
        correct = sum(1 for sf in step_feedback if sf.validity == StepValidity.CORRECT)
        partial = sum(1 for sf in step_feedback if sf.validity == StepValidity.PARTIAL)
        incorrect = sum(1 for sf in step_feedback if sf.validity == StepValidity.INCORRECT)
        total = len(step_feedback)

        pct = (
            (request.assigned_marks / request.total_marks * 100)
            if request.total_marks > 0
            else 0
        )

        summary = (
            f"You scored {request.assigned_marks:.1f}/{request.total_marks:.1f} marks "
            f"({pct:.0f}%) using the {request.detected_method} method. "
        )

        if total > 0:
            parts = []
            if correct:
                parts.append(f"{correct} step(s) fully correct")
            if partial:
                parts.append(f"{partial} partially correct")
            if incorrect:
                parts.append(f"{incorrect} incorrect")
            summary += ", ".join(parts) + "."

        if pct >= 80:
            summary += " Excellent work — keep it up!"
        elif pct >= 60:
            summary += " Good effort — review the highlighted steps to improve further."
        else:
            summary += " Focus on the highlighted steps to strengthen your understanding."

        return summary

    def _extract_suggestions(self, step_feedback: List[StepFeedback]) -> List[str]:
        suggestions = []
        for sf in step_feedback:
            if sf.validity != StepValidity.CORRECT and sf.how_to_improve:
                tip = sf.how_to_improve.strip()
                if tip and tip not in suggestions:
                    suggestions.append(tip)
        return suggestions[:3] or ["Review the incorrect steps and practice similar problems."]
