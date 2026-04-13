import os
from typing import List

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from app.models.schemas import FeedbackRequest, FeedbackResponse, StepFeedback


class FeedbackGenerator:
    """
    Core agent for Module 03 — Stepwise Feedback Generation.
    Uses Gemma 3n fine-tuned with LoRA to generate student-friendly feedback.
    """

    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._load_model()

    def _load_model(self):
        model_name = os.getenv("BASE_MODEL", "google/gemma-3n-E2B-it")
        lora_path = os.getenv("LORA_ADAPTER_PATH", None)

        print(f"[FeedbackGenerator] Loading {model_name} on {self.device}...")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto",
        )

        if lora_path and os.path.exists(lora_path):
            print(f"[FeedbackGenerator] Loading LoRA adapter from {lora_path}")
            self.model = PeftModel.from_pretrained(self.model, lora_path)

        self.model.eval()
        print("[FeedbackGenerator] Model ready.")

    def _build_prompt(self, request: FeedbackRequest) -> str:
        steps_text = "\n".join(
            [
                f"Step {s.step_number}: {s.expression} | "
                f"{'✓ Correct' if s.is_correct else '✗ Incorrect'} | "
                f"{s.marks_awarded} marks"
                + (f" | Error: {s.error_description}" if s.error_description else "")
                for s in request.student_steps
            ]
        )

        scheme_text = "\n".join(
            [
                f"Step {m.step_number}: {m.expected_expression} [{m.marks} marks]"
                for m in request.marking_scheme
            ]
        )

        return (
            f"<start_of_turn>user\n"
            f"You are an algebra teacher giving feedback on a student's exam answer.\n\n"
            f"Question: {request.question_text}\n"
            f"Solution Method: {request.detected_method}\n"
            f"Marks Awarded: {request.assigned_marks} / {request.total_marks}\n\n"
            f"Student's Steps:\n{steps_text}\n\n"
            f"Expected Marking Scheme:\n{scheme_text}\n\n"
            f"Give clear, step-by-step feedback that:\n"
            f"1. Confirms correct steps\n"
            f"2. Explains mistakes simply\n"
            f"3. Suggests improvements\n"
            f"Keep the language simple and encouraging for a student.\n"
            f"<end_of_turn>\n"
            f"<start_of_turn>model\n"
        )

    async def generate(self, request: FeedbackRequest) -> FeedbackResponse:
        prompt = self._build_prompt(request)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.7,
                do_sample=True,
                repetition_penalty=1.1,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
        feedback_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

        step_feedback = self._build_step_feedback(request.student_steps, feedback_text)
        suggestions = self._extract_suggestions(feedback_text)

        return FeedbackResponse(
            final_score=request.assigned_marks,
            total_marks=request.total_marks,
            step_feedback=step_feedback,
            overall_feedback=feedback_text,
            improvement_suggestions=suggestions,
        )

    def _build_step_feedback(self, steps, full_feedback: str) -> List[StepFeedback]:
        feedback_lines = [line.strip() for line in full_feedback.split("\n") if line.strip()]

        result = []
        for i, step in enumerate(steps):
            step_text = feedback_lines[i] if i < len(feedback_lines) else (
                "Good work!" if step.is_correct else f"Review Step {step.step_number}."
            )
            result.append(
                StepFeedback(
                    step_number=step.step_number,
                    expression=step.expression,
                    is_correct=step.is_correct,
                    marks_awarded=step.marks_awarded,
                    feedback=step_text,
                )
            )
        return result

    def _extract_suggestions(self, text: str) -> List[str]:
        suggestions = []
        keywords = ["improve", "suggest", "next time", "make sure", "remember", "try"]
        for line in text.split("\n"):
            line = line.strip()
            if any(kw in line.lower() for kw in keywords) and len(line) > 10:
                suggestions.append(line)
        return suggestions[:3] or ["Review the incorrect steps and practice similar problems."]
