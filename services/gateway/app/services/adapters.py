"""The schema bridge between the three modules.

Every function here is PURE — models in, models out, no I/O — so the riskiest
logic in the gateway is fully testable without a single service running.

Where the contracts actually disagree:

* OCR -> reasoning is nearly free. `{"reasoning_input": ...}` merged with
  `{"marking_scheme": ...}` *is* reasoning's EvaluationRequest.

* reasoning -> feedback is the real gap. reasoning emits `step_id`, a four-value
  `status`, and no `expression`; feedback wants `step_number`, a three-value
  `validity`, and an `expression` it renders verbatim into the SLM prompt.
  `build_feedback_request` closes that gap and records a warning every time it
  has to guess.
"""
from typing import Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from app.core.errors import MarkingSchemeInvalidError, NoStepsExtractedError
from app.schemas.upstream_feedback import FeedbackRequest, StepResult
from app.schemas.upstream_ocr import (
    OcrExtractResponse,
    OcrMarkingSchemeResponse,
    OcrStudentStep,
)
from app.schemas.upstream_reasoning import (
    EvaluationOutput,
    EvaluationRequest,
    MarkingScheme,
    ReasoningInput,
    SchemeStep,
    StudentStep,
)

#: Used when neither OCR nor the marking scheme can supply an expression.
#: Never send "" — feedback_generator.py renders `Step {n}: {expression}` straight
#: into the prompt, and a blank produces feedback about nothing.
UNKNOWN_EXPRESSION = "(expression not recognised)"

#: reasoning `status` -> feedback `validity`. "unclear" has no counterpart and is
#: resolved by _validity() below.
STATUS_MAP = {
    "correct": "correct",
    "incorrect": "incorrect",
    "partially_correct": "partial",
}


class NormalizedQuestion(BaseModel):
    """One question's worth of OCR output, cleaned up."""

    question_id: str
    question_text: str = ""
    student_steps: List[OcrStudentStep] = Field(default_factory=list)
    final_answer: Optional[str] = None


# ── OCR answer sheet ─────────────────────────────────────────────────────────

def normalize_ocr_questions(
    response: OcrExtractResponse, *, fallback_question_text: str = ""
) -> Tuple[List[NormalizedQuestion], List[str]]:
    """Flatten the singular/plural OCR shapes into one list of questions."""
    warnings: List[str] = []

    raw_questions = (
        response.reasoning_inputs
        if response.reasoning_inputs is not None
        else [response.reasoning_input]
    )

    questions: List[NormalizedQuestion] = []
    for index, raw in enumerate(raw_questions, start=1):
        if raw is None:
            continue

        question_id = (raw.question_id or "").strip() or f"Q{index}"

        kept = [step for step in raw.student_steps if step.content.strip()]
        if len(kept) != len(raw.student_steps):
            warnings.append(
                f"{question_id}: dropped {len(raw.student_steps) - len(kept)} "
                "empty step(s) returned by OCR."
            )

        # OCR already numbers steps 1..n densely; re-densifying is a canary for
        # that assumption breaking, since step_id is the join key used later.
        renumbered = [
            OcrStudentStep(step_id=position, content=step.content.strip())
            for position, step in enumerate(kept, start=1)
        ]
        if any(old.step_id != new.step_id for old, new in zip(kept, renumbered)):
            warnings.append(
                f"{question_id}: OCR step ids were not sequential and have been renumbered."
            )

        question_text = (raw.question_text or "").strip() or fallback_question_text.strip()
        if not question_text:
            warnings.append(
                f"{question_id}: no question text was extracted or supplied — "
                "grading quality will be reduced."
            )

        final_answer = (raw.final_answer or "").strip() or None

        questions.append(
            NormalizedQuestion(
                question_id=question_id,
                question_text=question_text,
                student_steps=renumbered,
                final_answer=final_answer,
            )
        )

    if not questions:
        raise NoStepsExtractedError(
            "OCR did not return any question from the uploaded answer sheet.",
            stage="ocr",
        )

    return questions, warnings


def select_questions(
    questions: List[NormalizedQuestion],
    *,
    policy: str,
    question_id: Optional[str] = None,
) -> Tuple[List[NormalizedQuestion], List[str]]:
    """Apply the multi-question policy.

    One uploaded marking scheme cannot correctly grade N different questions, so
    the default is to grade the first and say so out loud.
    """
    from app.core.errors import MultipleQuestionsError

    warnings: List[str] = []
    detected = len(questions)

    if question_id:
        chosen = [q for q in questions if q.question_id == question_id]
        if not chosen:
            available = ", ".join(q.question_id for q in questions)
            raise NoStepsExtractedError(
                f"No question with id '{question_id}' was found on the answer sheet. "
                f"Detected: {available}.",
                stage="ocr",
            )
        if detected > 1:
            warnings.append(
                f"OCR detected {detected} questions; only {question_id} was evaluated "
                "because a single marking scheme was supplied."
            )
        return chosen, warnings

    if detected == 1:
        return questions, warnings

    if policy == "error":
        raise MultipleQuestionsError(
            f"OCR detected {detected} questions on the answer sheet. Submit one question "
            "at a time, or pass question_id to choose one.",
            stage="ocr",
            details={"question_ids": [q.question_id for q in questions]},
        )

    if policy == "all":
        warnings.append(
            f"OCR detected {detected} questions; all were evaluated against the same "
            "marking scheme, so results for later questions may be unreliable."
        )
        return questions, warnings

    warnings.append(
        f"OCR detected {detected} questions; only {questions[0].question_id} was evaluated "
        "because a single marking scheme was supplied."
    )
    return questions[:1], warnings


# ── Marking scheme ───────────────────────────────────────────────────────────

def normalize_marking_scheme(
    response: OcrMarkingSchemeResponse,
) -> Tuple[MarkingScheme, List[str]]:
    """Clean and sanity-check the extracted scheme.

    This guards the worst silent failure in the whole pipeline: the OCR prompt
    says "if total marks are not visible, use 0", reasoning accepts total_marks=0
    (ge=0), and the results page then computes 0/0 and renders NaN%.
    """
    warnings: List[str] = []
    raw = response.marking_scheme

    kept = [
        step
        for step in raw.steps
        if (step.description or "").strip() or (step.expected_expression or "").strip()
    ]
    if len(kept) != len(raw.steps):
        warnings.append(
            f"Dropped {len(raw.steps) - len(kept)} blank marking-scheme step(s)."
        )
    if not kept:
        raise MarkingSchemeInvalidError(
            "No marking-scheme steps could be read from that image. Check the photo is "
            "sharp, upright, and shows the whole scheme.",
            stage="ocr",
        )

    steps = [
        SchemeStep(
            step_no=position,
            description=(step.description or "").strip()
            or (step.expected_expression or "").strip(),
            expected_expression=(step.expected_expression or "").strip()
            or (step.description or "").strip(),
            marks=max(0.0, float(step.marks or 0.0)),
        )
        for position, step in enumerate(kept, start=1)
    ]

    sum_marks = round(sum(step.marks for step in steps), 4)
    total = max(0.0, float(raw.total_marks or 0.0))

    if total <= 0 and sum_marks > 0:
        total = sum_marks
        warnings.append(
            f"The marking scheme image showed no total; using the sum of the step "
            f"marks ({sum_marks:g})."
        )
    elif total > 0 and sum_marks == 0:
        share = round(total / len(steps), 2)
        for step in steps[:-1]:
            step.marks = share
        steps[-1].marks = round(total - share * (len(steps) - 1), 2)
        warnings.append(
            f"No per-step marks were readable; the total of {total:g} was split evenly "
            f"across {len(steps)} steps."
        )
    elif total <= 0 and sum_marks == 0:
        raise MarkingSchemeInvalidError(
            "The marking scheme has no marks — neither a total nor any per-step values "
            "could be read from that image.",
            stage="ocr",
        )
    elif abs(total - sum_marks) > 0.01:
        warnings.append(
            f"The scheme total ({total:g}) does not match the sum of its step marks "
            f"({sum_marks:g}); using {total:g} as the maximum."
        )

    return MarkingScheme(total_marks=total, steps=steps), warnings


# ── OCR -> reasoning ─────────────────────────────────────────────────────────

def build_evaluation_request(
    question: NormalizedQuestion, scheme: MarkingScheme
) -> EvaluationRequest:
    if not question.student_steps:
        raise NoStepsExtractedError(
            f"No working steps were extracted for {question.question_id}. Check the "
            "answer sheet photo is legible.",
            stage="ocr",
        )

    return EvaluationRequest(
        reasoning_input=ReasoningInput(
            question_text=question.question_text,
            student_steps=[
                StudentStep(step_id=step.step_id, content=step.content)
                for step in question.student_steps
            ],
            final_answer=question.final_answer,
        ),
        marking_scheme=scheme,
    )


# ── reasoning -> feedback ────────────────────────────────────────────────────

def _validity(status: Optional[str], validity_flag: Optional[bool], marks_awarded: float) -> str:
    """Map reasoning's four-value status onto feedback's three-value enum."""
    normalized = (status or "").strip().lower()

    if normalized in STATUS_MAP:
        return STATUS_MAP[normalized]

    if normalized == "unclear":
        # feedback-service has no "unclear". Marks-aware: if the supervisor still
        # awarded credit the step earned something, so "partial" is the honest
        # reading; otherwise it earned nothing.
        return "partial" if marks_awarded > 0 else "incorrect"

    # Unknown or missing status — fall back to StepAnalysis.validity (a bool).
    if validity_flag is True:
        return "correct"
    if validity_flag is False:
        return "incorrect"
    return "incorrect"


def detect_reasoning_fallback(evaluation: EvaluationOutput) -> Optional[str]:
    """Spot reasoning's deterministic fallback path.

    When the supervisor LLM exhausts its retries, reasoning still answers HTTP
    200 — with confidence 0.3 everywhere and "fallback logic" in the summary.
    Marks are then a crude heuristic, which is worth surfacing.
    """
    if "fallback" in (evaluation.summary or "").lower():
        return "The reasoning service used its deterministic fallback; marks may be less reliable."

    analyses = evaluation.steps_analysis
    if analyses and all(
        step.confidence is not None and abs(step.confidence - 0.3) < 1e-6 for step in analyses
    ):
        return "The reasoning service reported uniformly low confidence; marks may be less reliable."

    return None


def build_feedback_request(
    question: NormalizedQuestion,
    scheme: MarkingScheme,
    evaluation: EvaluationOutput,
    *,
    include_unanalyzed_steps: bool = True,
) -> Tuple[FeedbackRequest, List[str]]:
    """Translate a reasoning result into a feedback request.

    reasoning never echoes back `question_text` or `marking_scheme`, so the
    gateway is their only holder at this point in the chain.
    """
    warnings: List[str] = []

    expression_by_step = {step.step_id: step.content for step in question.student_steps}
    scheme_expression_by_no = {step.step_no: step.expected_expression for step in scheme.steps}

    error_by_step: Dict[int, str] = {}
    if evaluation.step_validation:
        for validation in evaluation.step_validation.step_validations:
            if validation.error:
                error_by_step[validation.step_id] = validation.error

    missing_by_step: Dict[int, List[str]] = {}
    if evaluation.scheme_matching:
        for match in evaluation.scheme_matching.step_matches:
            if match.missing_elements:
                missing_by_step[match.step_id] = match.missing_elements

    by_number: Dict[int, StepResult] = {}

    for analysis in evaluation.steps_analysis:
        step_number = analysis.step_id

        expression = expression_by_step.get(step_number)
        if not expression:
            expression = scheme_expression_by_no.get(analysis.matched_scheme_step or -1)
            if expression:
                warnings.append(
                    f"Step {step_number}: no OCR text was available; showing the expected "
                    "expression from the marking scheme instead."
                )
            else:
                expression = UNKNOWN_EXPRESSION
                warnings.append(
                    f"Step {step_number}: the reasoning service reported a step that OCR "
                    "never extracted."
                )

        marks_awarded = float(analysis.marks_awarded or 0.0)
        if marks_awarded != marks_awarded:  # NaN
            marks_awarded = 0.0
        marks_awarded = round(marks_awarded, 2)

        error_description = error_by_step.get(step_number)
        if not error_description and missing_by_step.get(step_number):
            error_description = "Missing: " + ", ".join(missing_by_step[step_number])

        candidate = StepResult(
            step_number=step_number,
            expression=expression,
            validity=_validity(analysis.status, analysis.validity, marks_awarded),
            error_description=error_description,
            marks_awarded=marks_awarded,
        )

        existing = by_number.get(step_number)
        if existing is None:
            by_number[step_number] = candidate
        else:
            # The supervisor zeroes all but the best row within a scheme-step
            # group, so the highest-scoring duplicate is the representative one.
            warnings.append(
                f"Step {step_number}: the reasoning service returned it more than once; "
                "kept the highest-scoring entry."
            )
            if candidate.marks_awarded > existing.marks_awarded:
                by_number[step_number] = candidate

    unanalyzed = [
        step for step in question.student_steps if step.step_id not in by_number
    ]
    if unanalyzed:
        ids = ", ".join(str(step.step_id) for step in unanalyzed)
        if include_unanalyzed_steps:
            warnings.append(
                f"Step(s) {ids} were extracted from the answer but never analysed by the "
                "reasoning service; they are shown as unmatched."
            )
            for step in unanalyzed:
                by_number[step.step_id] = StepResult(
                    step_number=step.step_id,
                    expression=step.content or UNKNOWN_EXPRESSION,
                    validity="incorrect",
                    error_description="This step was not matched to any marking-scheme step.",
                    marks_awarded=0.0,
                )
        else:
            warnings.append(
                f"Step(s) {ids} were extracted from the answer but never analysed by the "
                "reasoning service; they have been omitted."
            )

    student_steps = [by_number[key] for key in sorted(by_number)]
    if not student_steps:
        raise NoStepsExtractedError(
            "Neither OCR nor the reasoning service produced any step to give feedback on.",
            stage="reasoning",
        )

    detected_method = ""
    if evaluation.method_detection:
        detected_method = (evaluation.method_detection.detected_method or "").strip()
    if not detected_method:
        detected_method = next(
            ((a.method or "").strip() for a in evaluation.steps_analysis if (a.method or "").strip()),
            "",
        )
    if not detected_method:
        detected_method = "undetermined"

    # reasoning's `total_marks` is marks EARNED; feedback's marking_scheme.total_marks
    # is marks AVAILABLE. Same name, opposite meaning.
    assigned_marks = round(min(max(float(evaluation.total_marks or 0.0), 0.0), scheme.total_marks), 2)

    request = FeedbackRequest(
        question_text=question.question_text,
        student_steps=student_steps,
        detected_method=detected_method,
        assigned_marks=assigned_marks,
        marking_scheme=scheme,
    )
    return request, warnings
