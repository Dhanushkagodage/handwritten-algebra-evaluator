"""Adapter tests — no network, no services, no credentials.

These cover the riskiest code in the gateway: the translation between three
contracts that were designed independently.
"""
import pytest

from app.core.errors import (
    MarkingSchemeInvalidError,
    MultipleQuestionsError,
    NoStepsExtractedError,
)
from app.schemas.upstream_ocr import OcrExtractResponse, OcrMarkingSchemeResponse
from app.schemas.upstream_reasoning import EvaluationOutput
from app.services.adapters import (
    UNKNOWN_EXPRESSION,
    NormalizedQuestion,
    build_evaluation_request,
    build_feedback_request,
    detect_reasoning_fallback,
    normalize_marking_scheme,
    normalize_ocr_questions,
    select_questions,
)


# ── Fixtures ─────────────────────────────────────────────────────────────────

def ocr_single(**overrides):
    payload = {
        "reasoning_input": {
            "question_text": "Solve x^2 - 5x + 6 = 0",
            "student_steps": [
                {"step_id": 1, "content": "x^2 - 5x + 6 = 0"},
                {"step_id": 2, "content": "(x - 2)(x - 3) = 0"},
                {"step_id": 3, "content": "x = 2 or x = 3"},
            ],
            "final_answer": "x = 2, x = 3",
        }
    }
    payload["reasoning_input"].update(overrides)
    return OcrExtractResponse.model_validate(payload)


def scheme_response(total_marks=6.0, marks=(2.0, 2.0, 2.0)):
    return OcrMarkingSchemeResponse.model_validate(
        {
            "marking_scheme": {
                "total_marks": total_marks,
                "steps": [
                    {
                        "step_no": index,
                        "description": f"Step {index}",
                        "expected_expression": f"expr{index}",
                        "marks": value,
                    }
                    for index, value in enumerate(marks, start=1)
                ],
            }
        }
    )


def question(steps=((1, "x^2 - 5x + 6 = 0"), (2, "(x - 2)(x - 3) = 0"))):
    return NormalizedQuestion(
        question_id="Q1",
        question_text="Solve x^2 - 5x + 6 = 0",
        student_steps=[{"step_id": sid, "content": text} for sid, text in steps],
    )


def evaluation(steps_analysis, **overrides):
    payload = {"steps_analysis": steps_analysis, "total_marks": 4.0, "max_marks": 6.0}
    payload.update(overrides)
    return EvaluationOutput.model_validate(payload)


def analysis(step_id, status="correct", marks=2.0, **overrides):
    row = {
        "step_id": step_id,
        "status": status,
        "validity": status == "correct",
        "marks_awarded": marks,
        "max_marks": 2.0,
        "method": "factorisation",
    }
    row.update(overrides)
    return row


# ── OCR normalisation ────────────────────────────────────────────────────────

def test_singular_reasoning_input_becomes_one_question():
    questions, warnings = normalize_ocr_questions(ocr_single())
    assert len(questions) == 1
    assert questions[0].question_id == "Q1"
    assert [s.step_id for s in questions[0].student_steps] == [1, 2, 3]
    assert warnings == []


def test_blank_steps_are_dropped_and_renumbered():
    response = ocr_single(
        student_steps=[
            {"step_id": 1, "content": "x = 1"},
            {"step_id": 2, "content": "   "},
            {"step_id": 3, "content": "x = 2"},
        ]
    )
    questions, warnings = normalize_ocr_questions(response)
    steps = questions[0].student_steps
    assert [s.step_id for s in steps] == [1, 2]
    assert [s.content for s in steps] == ["x = 1", "x = 2"]
    assert any("empty step" in w for w in warnings)


def test_missing_question_text_falls_back_to_the_supplied_one():
    response = ocr_single(question_text="")
    questions, _ = normalize_ocr_questions(response, fallback_question_text="Solve for x")
    assert questions[0].question_text == "Solve for x"


def test_plural_reasoning_inputs_are_all_returned():
    response = OcrExtractResponse.model_validate(
        {
            "reasoning_inputs": [
                {"question_id": "Q1", "question_text": "a", "student_steps": [{"step_id": 1, "content": "x"}]},
                {"question_id": "Q2", "question_text": "b", "student_steps": [{"step_id": 1, "content": "y"}]},
                {"question_id": "Q3", "question_text": "c", "student_steps": [{"step_id": 1, "content": "z"}]},
            ]
        }
    )
    questions, _ = normalize_ocr_questions(response)
    assert [q.question_id for q in questions] == ["Q1", "Q2", "Q3"]


def test_ocr_response_with_neither_key_is_rejected():
    with pytest.raises(ValueError):
        OcrExtractResponse.model_validate({})


# ── Multi-question policy ────────────────────────────────────────────────────

def three_questions():
    return normalize_ocr_questions(
        OcrExtractResponse.model_validate(
            {
                "reasoning_inputs": [
                    {"question_id": f"Q{n}", "question_text": "q", "student_steps": [{"step_id": 1, "content": "x"}]}
                    for n in (1, 2, 3)
                ]
            }
        )
    )[0]


def test_policy_first_grades_one_and_warns():
    selected, warnings = select_questions(three_questions(), policy="first")
    assert [q.question_id for q in selected] == ["Q1"]
    assert any("detected 3 questions" in w for w in warnings)


def test_policy_all_grades_every_question_and_warns():
    selected, warnings = select_questions(three_questions(), policy="all")
    assert len(selected) == 3
    assert any("same marking scheme" in w for w in warnings)


def test_policy_error_rejects():
    with pytest.raises(MultipleQuestionsError):
        select_questions(three_questions(), policy="error")


def test_explicit_question_id_selects_it():
    selected, warnings = select_questions(three_questions(), policy="first", question_id="Q2")
    assert [q.question_id for q in selected] == ["Q2"]
    assert any("only Q2 was evaluated" in w for w in warnings)


def test_unknown_question_id_is_rejected():
    with pytest.raises(NoStepsExtractedError):
        select_questions(three_questions(), policy="first", question_id="Q9")


def test_single_question_never_warns():
    selected, warnings = select_questions(normalize_ocr_questions(ocr_single())[0], policy="first")
    assert len(selected) == 1
    assert warnings == []


# ── Marking scheme normalisation ─────────────────────────────────────────────

def test_healthy_scheme_passes_through_without_warnings():
    scheme, warnings = normalize_marking_scheme(scheme_response())
    assert scheme.total_marks == 6.0
    assert [s.marks for s in scheme.steps] == [2.0, 2.0, 2.0]
    assert warnings == []


def test_zero_total_is_backfilled_from_the_step_marks():
    """The OCR prompt emits total_marks=0 when no total is visible; unguarded
    that reaches the UI as a 0/0 division and renders NaN%."""
    scheme, warnings = normalize_marking_scheme(scheme_response(total_marks=0.0))
    assert scheme.total_marks == 6.0
    assert any("showed no total" in w for w in warnings)


def test_zero_step_marks_are_distributed_from_the_total():
    scheme, warnings = normalize_marking_scheme(
        scheme_response(total_marks=5.0, marks=(0.0, 0.0))
    )
    assert sum(s.marks for s in scheme.steps) == pytest.approx(5.0)
    assert any("split evenly" in w for w in warnings)


def test_scheme_with_no_marks_at_all_is_rejected():
    with pytest.raises(MarkingSchemeInvalidError):
        normalize_marking_scheme(scheme_response(total_marks=0.0, marks=(0.0, 0.0)))


def test_scheme_with_no_readable_steps_is_rejected():
    response = OcrMarkingSchemeResponse.model_validate(
        {"marking_scheme": {"total_marks": 5.0, "steps": []}}
    )
    with pytest.raises(MarkingSchemeInvalidError):
        normalize_marking_scheme(response)


def test_total_disagreeing_with_step_sum_warns_but_keeps_the_total():
    scheme, warnings = normalize_marking_scheme(
        scheme_response(total_marks=10.0, marks=(2.0, 2.0, 2.0))
    )
    assert scheme.total_marks == 10.0
    assert any("does not match" in w for w in warnings)


# ── OCR -> reasoning ─────────────────────────────────────────────────────────

def test_evaluation_request_is_a_straight_merge():
    scheme, _ = normalize_marking_scheme(scheme_response())
    request = build_evaluation_request(question(), scheme)
    assert request.reasoning_input.question_text == "Solve x^2 - 5x + 6 = 0"
    assert [s.step_id for s in request.reasoning_input.student_steps] == [1, 2]
    assert request.marking_scheme.total_marks == 6.0


def test_question_with_no_steps_is_rejected():
    scheme, _ = normalize_marking_scheme(scheme_response())
    empty = NormalizedQuestion(question_id="Q1", question_text="q", student_steps=[])
    with pytest.raises(NoStepsExtractedError):
        build_evaluation_request(empty, scheme)


# ── reasoning -> feedback: the status mapping ────────────────────────────────

@pytest.mark.parametrize(
    "status,marks,expected",
    [
        ("correct", 2.0, "correct"),
        ("incorrect", 0.0, "incorrect"),
        ("partially_correct", 1.0, "partial"),
        # "unclear" has no counterpart in feedback's three-value enum, so it is
        # resolved by whether the supervisor still awarded credit.
        ("unclear", 1.0, "partial"),
        ("unclear", 0.0, "incorrect"),
        ("PARTIALLY_CORRECT", 1.0, "partial"),
        ("  correct  ", 2.0, "correct"),
    ],
)
def test_status_maps_onto_feedback_validity(status, marks, expected):
    scheme, _ = normalize_marking_scheme(scheme_response())
    request, _ = build_feedback_request(
        question(), scheme, evaluation([analysis(1, status=status, marks=marks)])
    )
    assert request.student_steps[0].validity == expected


def test_unknown_status_falls_back_to_the_boolean_validity_flag():
    scheme, _ = normalize_marking_scheme(scheme_response())
    request, _ = build_feedback_request(
        question(),
        scheme,
        evaluation([analysis(1, status="weird", marks=2.0, validity=True)]),
    )
    assert request.student_steps[0].validity == "correct"


# ── reasoning -> feedback: the expression join ───────────────────────────────

def test_expression_is_joined_from_ocr_by_step_id():
    scheme, _ = normalize_marking_scheme(scheme_response())
    request, warnings = build_feedback_request(
        question(), scheme, evaluation([analysis(1), analysis(2)])
    )
    assert [s.expression for s in request.student_steps] == [
        "x^2 - 5x + 6 = 0",
        "(x - 2)(x - 3) = 0",
    ]
    assert warnings == []


def test_step_missing_from_ocr_falls_back_to_the_matched_scheme_expression():
    scheme, _ = normalize_marking_scheme(scheme_response())
    request, warnings = build_feedback_request(
        question(steps=((1, "x = 1"),)),
        scheme,
        evaluation([analysis(1), analysis(7, matched_scheme_step=2)]),
    )
    step7 = next(s for s in request.student_steps if s.step_number == 7)
    assert step7.expression == "expr2"
    assert any("marking scheme instead" in w for w in warnings)


def test_step_with_no_source_at_all_gets_a_placeholder_never_a_blank():
    """feedback_generator renders `Step {n}: {expression}` verbatim into the
    prompt, so an empty string produces feedback about nothing."""
    scheme, _ = normalize_marking_scheme(scheme_response())
    request, warnings = build_feedback_request(
        question(steps=((1, "x = 1"),)), scheme, evaluation([analysis(9)])
    )
    step9 = next(s for s in request.student_steps if s.step_number == 9)
    assert step9.expression == UNKNOWN_EXPRESSION
    assert any("OCR never extracted" in w for w in warnings)


# ── reasoning -> feedback: step reconciliation ───────────────────────────────

def test_unanalyzed_ocr_steps_are_kept_by_default():
    scheme, _ = normalize_marking_scheme(scheme_response())
    request, warnings = build_feedback_request(
        question(steps=((1, "a"), (2, "b"), (3, "c"))), scheme, evaluation([analysis(1)])
    )
    assert [s.step_number for s in request.student_steps] == [1, 2, 3]
    unmatched = request.student_steps[1]
    assert unmatched.validity == "incorrect"
    assert unmatched.marks_awarded == 0.0
    assert "not matched" in (unmatched.error_description or "")
    assert any("never analysed" in w for w in warnings)


def test_unanalyzed_ocr_steps_can_be_dropped():
    scheme, _ = normalize_marking_scheme(scheme_response())
    request, warnings = build_feedback_request(
        question(steps=((1, "a"), (2, "b"))),
        scheme,
        evaluation([analysis(1)]),
        include_unanalyzed_steps=False,
    )
    assert [s.step_number for s in request.student_steps] == [1]
    assert any("omitted" in w for w in warnings)


def test_duplicate_step_ids_are_deduped_keeping_the_highest_marks():
    """The supervisor zeroes all but the best row within a scheme-step group."""
    scheme, _ = normalize_marking_scheme(scheme_response())
    request, warnings = build_feedback_request(
        question(steps=((1, "a"),)),
        scheme,
        evaluation([analysis(1, marks=0.0), analysis(1, marks=2.0)]),
    )
    assert len(request.student_steps) == 1
    assert request.student_steps[0].marks_awarded == 2.0
    assert any("more than once" in w for w in warnings)


def test_steps_are_sorted_by_number():
    scheme, _ = normalize_marking_scheme(scheme_response())
    request, _ = build_feedback_request(
        question(steps=((1, "a"), (2, "b"), (3, "c"))),
        scheme,
        evaluation([analysis(3), analysis(1), analysis(2)]),
    )
    assert [s.step_number for s in request.student_steps] == [1, 2, 3]


def test_no_steps_anywhere_is_rejected():
    scheme, _ = normalize_marking_scheme(scheme_response())
    empty = NormalizedQuestion(question_id="Q1", question_text="q", student_steps=[])
    with pytest.raises(NoStepsExtractedError):
        build_feedback_request(empty, scheme, evaluation([]))


# ── reasoning -> feedback: the surrounding fields ────────────────────────────

def test_error_description_comes_from_step_validation():
    scheme, _ = normalize_marking_scheme(scheme_response())
    request, _ = build_feedback_request(
        question(steps=((1, "a"),)),
        scheme,
        evaluation(
            [analysis(1, status="incorrect", marks=0.0)],
            step_validation={
                "step_validations": [
                    {"step_id": 1, "is_valid": False, "status": "incorrect", "error": "Sign error"}
                ]
            },
        ),
    )
    assert request.student_steps[0].error_description == "Sign error"


def test_error_description_falls_back_to_scheme_matching_gaps():
    scheme, _ = normalize_marking_scheme(scheme_response())
    request, _ = build_feedback_request(
        question(steps=((1, "a"),)),
        scheme,
        evaluation(
            [analysis(1, status="partially_correct", marks=1.0)],
            scheme_matching={
                "step_matches": [
                    {"step_id": 1, "matched_scheme_step": 1, "missing_elements": ["factorisation", "roots"]}
                ]
            },
        ),
    )
    assert request.student_steps[0].error_description == "Missing: factorisation, roots"


def test_null_step_validation_does_not_crash():
    scheme, _ = normalize_marking_scheme(scheme_response())
    request, _ = build_feedback_request(
        question(steps=((1, "a"),)), scheme, evaluation([analysis(1)], step_validation=None)
    )
    assert request.student_steps[0].error_description is None


def test_detected_method_prefers_the_method_detection_agent():
    scheme, _ = normalize_marking_scheme(scheme_response())
    request, _ = build_feedback_request(
        question(steps=((1, "a"),)),
        scheme,
        evaluation([analysis(1)], method_detection={"detected_method": "quadratic formula"}),
    )
    assert request.detected_method == "quadratic formula"


def test_detected_method_falls_back_to_the_per_step_method():
    scheme, _ = normalize_marking_scheme(scheme_response())
    request, _ = build_feedback_request(
        question(steps=((1, "a"),)), scheme, evaluation([analysis(1)])
    )
    assert request.detected_method == "factorisation"


def test_detected_method_defaults_to_undetermined():
    scheme, _ = normalize_marking_scheme(scheme_response())
    request, _ = build_feedback_request(
        question(steps=((1, "a"),)), scheme, evaluation([analysis(1, method="")])
    )
    assert request.detected_method == "undetermined"


def test_assigned_marks_come_from_reasoning_total_and_are_clamped():
    """reasoning's total_marks means marks EARNED; the scheme's means marks
    AVAILABLE. Same name, opposite meaning."""
    scheme, _ = normalize_marking_scheme(scheme_response())
    request, _ = build_feedback_request(
        question(steps=((1, "a"),)), scheme, evaluation([analysis(1)], total_marks=99.0)
    )
    assert request.assigned_marks == 6.0

    request, _ = build_feedback_request(
        question(steps=((1, "a"),)), scheme, evaluation([analysis(1)], total_marks=-3.0)
    )
    assert request.assigned_marks == 0.0


def test_question_text_and_scheme_are_carried_by_the_gateway():
    """reasoning echoes back neither, so the gateway is their only holder."""
    scheme, _ = normalize_marking_scheme(scheme_response())
    request, _ = build_feedback_request(
        question(steps=((1, "a"),)), scheme, evaluation([analysis(1)])
    )
    assert request.question_text == "Solve x^2 - 5x + 6 = 0"
    assert request.marking_scheme.total_marks == 6.0
    assert len(request.marking_scheme.steps) == 3


# ── Degraded-reasoning detection ─────────────────────────────────────────────

def test_fallback_summary_is_detected():
    warning = detect_reasoning_fallback(
        evaluation([analysis(1)], summary="Generated with fallback logic.")
    )
    assert warning and "fallback" in warning


def test_uniform_low_confidence_is_detected():
    warning = detect_reasoning_fallback(
        evaluation([analysis(1, confidence=0.3), analysis(2, confidence=0.3)])
    )
    assert warning and "low confidence" in warning


def test_healthy_evaluation_produces_no_warning():
    assert (
        detect_reasoning_fallback(
            evaluation([analysis(1, confidence=0.95)], summary="Well structured solution.")
        )
        is None
    )
