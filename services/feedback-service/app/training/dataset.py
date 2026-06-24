"""
Dataset preparation for LoRA fine-tuning of the feedback generation module.

Raw annotation schema (raw_annotations.json):
[
  {
    "question": "Solve x^2 - 5x + 6 = 0",
    "method": "factorisation",
    "assigned_marks": 4.0,
    "total_marks": 4.0,
    "steps": [
      {
        "step_number": 1,
        "expression": "x^2 - 5x + 6 = 0",
        "validity": "correct",          // "correct" | "partial" | "incorrect"
        "marks_awarded": 1.0,
        "error_description": null
      }
    ],
    "marking_scheme": [
      {
        "step_number": 1,
        "expected_expression": "x^2 - 5x + 6 = 0",
        "marks": 1.0,
        "description": "State the equation"
      }
    ],
    "step_feedback": [
      {
        "step_number": 1,
        "correct": "You correctly identified the quadratic equation.",
        "missing": null,      // only for partial / incorrect
        "deduction": null,    // only for partial / incorrect
        "improve": "Well done, continue to the next step."
      }
    ]
  }
]
"""

import json
import os
from typing import Dict, List

# Must stay in sync with app/agents/feedback_generator.py _FORMAT_INSTRUCTION
_FORMAT_INSTRUCTION = (
    "\nFor each step, respond in this exact format:\n"
    "=== STEP N [CORRECT/PARTIAL/INCORRECT] ===\n"
    "CORRECT: <what the student did correctly, or method acknowledgement>\n"
    "MISSING: <what was wrong or missing — only for INCORRECT or PARTIAL>\n"
    "DEDUCTION: <why marks were reduced — only for INCORRECT or PARTIAL>\n"
    "IMPROVE: <specific actionable tip for the student>\n"
)

# Qwen2.5 ChatML special tokens — must match feedback_generator.py
_IM_START = "<|im_start|>"
_IM_END = "<|im_end|>"


def _build_target(item: Dict) -> str:
    """Build the structured === STEP N === completion from step_feedback annotations."""
    feedback_map = {fb["step_number"]: fb for fb in item["step_feedback"]}
    blocks = []
    for step in item["steps"]:
        n = step["step_number"]
        validity = step["validity"].upper()
        fb = feedback_map.get(n, {})

        lines = [f"=== STEP {n} [{validity}] ==="]
        lines.append(f"CORRECT: {fb.get('correct', 'Step acknowledged.')}")
        if step["validity"] in ("partial", "incorrect"):
            missing = fb.get("missing")
            deduction = fb.get("deduction")
            if missing:
                lines.append(f"MISSING: {missing}")
            if deduction:
                lines.append(f"DEDUCTION: {deduction}")
        lines.append(f"IMPROVE: {fb.get('improve', 'Review this step.')}")
        blocks.append("\n".join(lines))

    return "\n\n".join(blocks)


def format_example(item: Dict) -> Dict:
    """Format a raw annotation into the prompt-completion pair for SFT training."""
    scheme_marks = {m["step_number"]: m["marks"] for m in item["marking_scheme"]}

    steps_text = "\n".join(
        "Step {n}: {expr} [{v}, {awarded}/{possible} marks]{err}".format(
            n=s["step_number"],
            expr=s["expression"],
            v=s["validity"].upper(),
            awarded=s["marks_awarded"],
            possible=scheme_marks.get(s["step_number"], "?"),
            err=f" — Error: {s['error_description']}" if s.get("error_description") else "",
        )
        for s in item["steps"]
    )

    scheme_text = "\n".join(
        "Step {n}: {expr} [{m} marks]{desc}".format(
            n=m["step_number"],
            expr=m["expected_expression"],
            m=m["marks"],
            desc=f" — {m['description']}" if m.get("description") else "",
        )
        for m in item["marking_scheme"]
    )

    prompt = (
        f"{_IM_START}system\n"
        f"You are an algebra teacher giving feedback on a student's exam answer.{_IM_END}\n"
        f"{_IM_START}user\n"
        f"Question: {item['question']}\n"
        f"Solution Method: {item['method']}\n"
        f"Score: {item['assigned_marks']} / {item['total_marks']}\n\n"
        f"Student's Steps:\n{steps_text}\n\n"
        f"Marking Scheme:\n{scheme_text}\n"
        f"{_FORMAT_INSTRUCTION}"
        f"{_IM_END}\n"
        f"{_IM_START}assistant\n"
    )

    target = _build_target(item)
    return {"text": f"{prompt}{target}{_IM_END}"}


def prepare_dataset(input_file: str, output_file: str) -> List[Dict]:
    """Convert raw annotations to training-ready JSON format."""
    with open(input_file, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    formatted = [format_example(item) for item in raw_data]

    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(formatted, f, indent=2, ensure_ascii=False)

    print(f"Prepared {len(formatted)} examples → {output_file}")
    return formatted


def create_sample_annotations(output_file: str):
    """
    Create starter raw annotations covering all 5 error families:
      1. Sign errors
      2. Missing intermediate steps
      3. Incomplete simplification
      4. Incorrect factorisation
      5. Correct method with erroneous final result
    Plus fully correct examples for balanced training.
    """
    samples = [
        # ── Error family 1: Sign error in factorisation result ────────────────
        {
            "question": "Solve x^2 + x - 6 = 0",
            "method": "factorisation",
            "assigned_marks": 3.5,
            "total_marks": 4.0,
            "steps": [
                {
                    "step_number": 1,
                    "expression": "x^2 + x - 6 = 0",
                    "validity": "correct",
                    "marks_awarded": 1.0,
                    "error_description": None,
                },
                {
                    "step_number": 2,
                    "expression": "(x + 3)(x - 2) = 0",
                    "validity": "correct",
                    "marks_awarded": 2.0,
                    "error_description": None,
                },
                {
                    "step_number": 3,
                    "expression": "x = -3 or x = -2",
                    "validity": "incorrect",
                    "marks_awarded": 0.5,
                    "error_description": "Sign error: (x - 2) = 0 gives x = +2, not x = -2",
                },
            ],
            "marking_scheme": [
                {"step_number": 1, "expected_expression": "x^2 + x - 6 = 0", "marks": 1.0, "description": "State the equation"},
                {"step_number": 2, "expected_expression": "(x + 3)(x - 2) = 0", "marks": 2.0, "description": "Correct factorisation"},
                {"step_number": 3, "expected_expression": "x = -3 or x = 2", "marks": 1.0, "description": "Both roots correct"},
            ],
            "step_feedback": [
                {
                    "step_number": 1,
                    "correct": "You correctly identified and stated the quadratic equation.",
                    "missing": None,
                    "deduction": None,
                    "improve": "Well done, continue to the next step.",
                },
                {
                    "step_number": 2,
                    "correct": "Excellent — your factorisation (x + 3)(x - 2) = 0 is completely correct.",
                    "missing": None,
                    "deduction": None,
                    "improve": "Well done, continue to the next step.",
                },
                {
                    "step_number": 3,
                    "correct": "You applied the zero-product property and correctly found x = -3 from (x + 3) = 0.",
                    "missing": "The second root is wrong. From (x - 2) = 0, adding 2 to both sides gives x = +2, not x = -2.",
                    "deduction": "Half a mark awarded for method; half lost because x = -2 has the wrong sign — the correct second root is x = +2.",
                    "improve": "When solving (x - a) = 0, the root is x = +a, not -a. Always verify by substituting back: (-2)^2 + (-2) - 6 = -4 ≠ 0, confirming x = -2 is wrong.",
                },
            ],
        },
        # ── Error family 2: Missing second root ───────────────────────────────
        {
            "question": "Solve x^2 - 9 = 0",
            "method": "difference of squares",
            "assigned_marks": 2.0,
            "total_marks": 3.0,
            "steps": [
                {
                    "step_number": 1,
                    "expression": "(x - 3)(x + 3) = 0",
                    "validity": "correct",
                    "marks_awarded": 1.5,
                    "error_description": None,
                },
                {
                    "step_number": 2,
                    "expression": "x = 3",
                    "validity": "partial",
                    "marks_awarded": 0.5,
                    "error_description": "Missing second root x = -3",
                },
            ],
            "marking_scheme": [
                {"step_number": 1, "expected_expression": "(x - 3)(x + 3) = 0", "marks": 1.5, "description": "Factor using difference of squares"},
                {"step_number": 2, "expected_expression": "x = 3 or x = -3", "marks": 1.5, "description": "State both roots"},
            ],
            "step_feedback": [
                {
                    "step_number": 1,
                    "correct": "Perfect application of the difference-of-squares identity: a^2 - b^2 = (a - b)(a + b).",
                    "missing": None,
                    "deduction": None,
                    "improve": "Well done, continue to the next step.",
                },
                {
                    "step_number": 2,
                    "correct": "x = 3 is one correct root, obtained correctly from (x - 3) = 0.",
                    "missing": "The second root x = -3 is missing. From (x + 3) = 0, subtracting 3 from both sides gives x = -3.",
                    "deduction": "1 mark deducted because a factored quadratic has two roots — only one was stated.",
                    "improve": "After applying the zero-product property, solve each factor separately: (x - 3) = 0 → x = 3 and (x + 3) = 0 → x = -3. Write the final answer as x = 3 or x = -3.",
                },
            ],
        },
        # ── Error family 3: Incomplete simplification ─────────────────────────
        {
            "question": "Solve 4x - 12 = 0",
            "method": "linear equation",
            "assigned_marks": 1.5,
            "total_marks": 2.0,
            "steps": [
                {
                    "step_number": 1,
                    "expression": "4x = 12",
                    "validity": "correct",
                    "marks_awarded": 1.0,
                    "error_description": None,
                },
                {
                    "step_number": 2,
                    "expression": "x = 12/4",
                    "validity": "partial",
                    "marks_awarded": 0.5,
                    "error_description": "Answer left as unsimplified fraction — must evaluate to x = 3",
                },
            ],
            "marking_scheme": [
                {"step_number": 1, "expected_expression": "4x = 12", "marks": 1.0, "description": "Isolate the x term"},
                {"step_number": 2, "expected_expression": "x = 3", "marks": 1.0, "description": "Solve for x (simplified)"},
            ],
            "step_feedback": [
                {
                    "step_number": 1,
                    "correct": "You correctly added 12 to isolate the x term: 4x = 12.",
                    "missing": None,
                    "deduction": None,
                    "improve": "Well done, continue to the next step.",
                },
                {
                    "step_number": 2,
                    "correct": "Dividing both sides by 4 is the correct operation, and x = 12/4 shows the correct division step.",
                    "missing": "The fraction 12/4 must be evaluated — the final answer should be x = 3, not x = 12/4.",
                    "deduction": "Half a mark deducted because the answer was left as an unsimplified fraction; exam answers require a fully simplified numerical value.",
                    "improve": "Always compute the division: 12 ÷ 4 = 3, so write x = 3 as your final answer.",
                },
            ],
        },
        # ── Error family 3 (variant): Incomplete simplification in quadratic ──
        {
            "question": "Solve 2x^2 - 8 = 0",
            "method": "direct algebraic manipulation",
            "assigned_marks": 2.5,
            "total_marks": 4.0,
            "steps": [
                {
                    "step_number": 1,
                    "expression": "2x^2 = 8",
                    "validity": "correct",
                    "marks_awarded": 1.0,
                    "error_description": None,
                },
                {
                    "step_number": 2,
                    "expression": "x^2 = 4",
                    "validity": "correct",
                    "marks_awarded": 1.0,
                    "error_description": None,
                },
                {
                    "step_number": 3,
                    "expression": "x = 2",
                    "validity": "partial",
                    "marks_awarded": 0.5,
                    "error_description": "Missing negative root x = -2; x^2 = 4 has two solutions",
                },
            ],
            "marking_scheme": [
                {"step_number": 1, "expected_expression": "2x^2 = 8", "marks": 1.0, "description": "Rearrange to isolate x^2 term"},
                {"step_number": 2, "expected_expression": "x^2 = 4", "marks": 1.0, "description": "Divide both sides by 2"},
                {"step_number": 3, "expected_expression": "x = ±2", "marks": 2.0, "description": "Both positive and negative roots"},
            ],
            "step_feedback": [
                {
                    "step_number": 1,
                    "correct": "You correctly rearranged the equation by adding 8 to both sides.",
                    "missing": None,
                    "deduction": None,
                    "improve": "Well done, continue to the next step.",
                },
                {
                    "step_number": 2,
                    "correct": "Dividing both sides by 2 to get x^2 = 4 is correct.",
                    "missing": None,
                    "deduction": None,
                    "improve": "Well done, continue to the next step.",
                },
                {
                    "step_number": 3,
                    "correct": "x = 2 is one valid solution — 2^2 = 4 ✓.",
                    "missing": "The equation x^2 = 4 also has the solution x = -2, since (-2)^2 = 4. The complete answer is x = ±2.",
                    "deduction": "1.5 marks deducted for the missing negative root — square roots always yield two solutions (positive and negative) unless the context restricts to positive values.",
                    "improve": "When solving x^2 = k (k > 0), always write x = ±√k to capture both roots. Here x = ±√4 = ±2.",
                },
            ],
        },
        # ── Error family 4: Incorrect factorisation ───────────────────────────
        {
            "question": "Solve x^2 - 7x + 12 = 0",
            "method": "factorisation",
            "assigned_marks": 1.0,
            "total_marks": 4.0,
            "steps": [
                {
                    "step_number": 1,
                    "expression": "x^2 - 7x + 12 = 0",
                    "validity": "correct",
                    "marks_awarded": 1.0,
                    "error_description": None,
                },
                {
                    "step_number": 2,
                    "expression": "(x - 4)(x + 3) = 0",
                    "validity": "incorrect",
                    "marks_awarded": 0.0,
                    "error_description": "Wrong sign in second factor: (x + 3) expands to give +3x term, so product is x^2 - x - 12, not x^2 - 7x + 12",
                },
                {
                    "step_number": 3,
                    "expression": "x = 4 or x = -3",
                    "validity": "incorrect",
                    "marks_awarded": 0.0,
                    "error_description": "Roots follow from the incorrect factorisation in Step 2",
                },
            ],
            "marking_scheme": [
                {"step_number": 1, "expected_expression": "x^2 - 7x + 12 = 0", "marks": 1.0, "description": "State the equation"},
                {"step_number": 2, "expected_expression": "(x - 4)(x - 3) = 0", "marks": 2.0, "description": "Correct factorisation"},
                {"step_number": 3, "expected_expression": "x = 4 or x = 3", "marks": 1.0, "description": "Both roots correct"},
            ],
            "step_feedback": [
                {
                    "step_number": 1,
                    "correct": "You correctly stated the quadratic equation to be solved.",
                    "missing": None,
                    "deduction": None,
                    "improve": "Well done, continue to the next step.",
                },
                {
                    "step_number": 2,
                    "correct": "You identified the correct factorisation approach and found one correct factor (x - 4).",
                    "missing": "The second factor is wrong. You need two numbers that multiply to +12 AND add to -7. Those are -4 and -3, giving (x - 4)(x - 3). Expanding (x - 4)(x + 3) gives x^2 - x - 12, which does not match.",
                    "deduction": "2 marks deducted because (x - 4)(x + 3) = x^2 - x - 12 ≠ x^2 - 7x + 12.",
                    "improve": "To factorise x^2 + bx + c, list all factor pairs of c that sum to b. Here c = +12, b = -7: try (-4) × (-3) = 12 and (-4) + (-3) = -7 ✓. So the correct factorisation is (x - 4)(x - 3).",
                },
                {
                    "step_number": 3,
                    "correct": "You correctly applied the zero-product property to derive roots from your factors.",
                    "missing": "Because Step 2 was factorised incorrectly, the roots x = 4 and x = -3 are wrong. The correct roots (from (x - 4)(x - 3) = 0) are x = 4 and x = 3.",
                    "deduction": "1 mark deducted because the roots are based on an incorrect factorisation.",
                    "improve": "After correcting the factorisation to (x - 4)(x - 3) = 0: x - 4 = 0 → x = 4 and x - 3 = 0 → x = 3. Verify: 4^2 - 7(4) + 12 = 0 ✓ and 3^2 - 7(3) + 12 = 0 ✓.",
                },
            ],
        },
        # ── Error family 5: Correct method, arithmetic slip in final step ──────
        {
            "question": "Solve x^2 - 9x + 20 = 0",
            "method": "factorisation",
            "assigned_marks": 3.0,
            "total_marks": 4.0,
            "steps": [
                {
                    "step_number": 1,
                    "expression": "x^2 - 9x + 20 = 0",
                    "validity": "correct",
                    "marks_awarded": 1.0,
                    "error_description": None,
                },
                {
                    "step_number": 2,
                    "expression": "(x - 4)(x - 5) = 0",
                    "validity": "correct",
                    "marks_awarded": 2.0,
                    "error_description": None,
                },
                {
                    "step_number": 3,
                    "expression": "x = 4 or x = 6",
                    "validity": "incorrect",
                    "marks_awarded": 0.0,
                    "error_description": "Arithmetic slip: (x - 5) = 0 gives x = 5, not x = 6",
                },
            ],
            "marking_scheme": [
                {"step_number": 1, "expected_expression": "x^2 - 9x + 20 = 0", "marks": 1.0, "description": "State the equation"},
                {"step_number": 2, "expected_expression": "(x - 4)(x - 5) = 0", "marks": 2.0, "description": "Correct factorisation"},
                {"step_number": 3, "expected_expression": "x = 4 or x = 5", "marks": 1.0, "description": "Both roots correct"},
            ],
            "step_feedback": [
                {
                    "step_number": 1,
                    "correct": "You correctly stated the quadratic equation.",
                    "missing": None,
                    "deduction": None,
                    "improve": "Well done, continue to the next step.",
                },
                {
                    "step_number": 2,
                    "correct": "Excellent factorisation — (x - 4)(x - 5) is completely correct. You found the factor pair -4 and -5 that multiplies to +20 and adds to -9.",
                    "missing": None,
                    "deduction": None,
                    "improve": "Well done, continue to the next step.",
                },
                {
                    "step_number": 3,
                    "correct": "x = 4 is correct, obtained from (x - 4) = 0.",
                    "missing": "The second root is wrong. From (x - 5) = 0, adding 5 to both sides gives x = 5, not x = 6. This appears to be an arithmetic slip.",
                    "deduction": "1 mark deducted because x = 6 is incorrect — the correct second root is x = 5.",
                    "improve": "When reading a root from a factor (x - a) = 0, the answer is x = a. Here (x - 5) = 0 → x = 5. A quick check would catch the error: 6^2 - 9(6) + 20 = 2 ≠ 0, so x = 6 is clearly wrong.",
                },
            ],
        },
        # ── Error family 5 (variant): Correct method, wrong discriminant ──────
        {
            "question": "Solve x^2 + 5x + 6 = 0 using the quadratic formula",
            "method": "quadratic formula",
            "assigned_marks": 2.5,
            "total_marks": 5.0,
            "steps": [
                {
                    "step_number": 1,
                    "expression": "x = (5 ± √(25 - 24)) / 2",
                    "validity": "incorrect",
                    "marks_awarded": 0.5,
                    "error_description": "Sign error: formula requires -b; here b = +5, so -b = -5. Student wrote +5 instead of -5.",
                },
                {
                    "step_number": 2,
                    "expression": "x = (5 ± 1) / 2",
                    "validity": "correct",
                    "marks_awarded": 1.0,
                    "error_description": None,
                },
                {
                    "step_number": 3,
                    "expression": "x = 3 or x = 2",
                    "validity": "incorrect",
                    "marks_awarded": 1.0,
                    "error_description": "Roots are wrong because Step 1 used +5 instead of -5 in the numerator",
                },
            ],
            "marking_scheme": [
                {"step_number": 1, "expected_expression": "x = (-5 ± √(25 - 24)) / 2", "marks": 2.0, "description": "Apply quadratic formula with -b"},
                {"step_number": 2, "expected_expression": "x = (-5 ± 1) / 2", "marks": 1.0, "description": "Simplify discriminant"},
                {"step_number": 3, "expected_expression": "x = -2 or x = -3", "marks": 2.0, "description": "Both roots correct"},
            ],
            "step_feedback": [
                {
                    "step_number": 1,
                    "correct": "You correctly identified the discriminant as b^2 - 4ac = 25 - 24 = 1, and the denominator as 2a = 2.",
                    "missing": "The numerator uses the wrong sign. The quadratic formula is x = (-b ± √(b^2 - 4ac)) / 2a. Here b = +5, so -b = -5. You wrote +5 instead.",
                    "deduction": "1.5 marks deducted because using +b instead of -b in the numerator produces entirely wrong roots.",
                    "improve": "Memorise the formula as x = (−b ± √(b²−4ac)) / 2a — note the negative sign before b. For x^2 + 5x + 6: a=1, b=5, c=6, so the numerator is −5 ± √(25 − 24) = −5 ± 1.",
                },
                {
                    "step_number": 2,
                    "correct": "The simplification of the square root is correct: √1 = 1, giving (5 ± 1)/2 — consistent with your Step 1.",
                    "missing": None,
                    "deduction": None,
                    "improve": "The arithmetic here is correct; correct the sign in Step 1 and this step follows naturally.",
                },
                {
                    "step_number": 3,
                    "correct": "You correctly evaluated the two ± cases from the expression in Step 2.",
                    "missing": "Because Step 1 had the wrong sign, x = 3 and x = 2 are incorrect. The correct roots (using -5 ± 1)/2 are x = -2 and x = -3.",
                    "deduction": "1 mark deducted because the roots follow from the sign error in Step 1; method was applied correctly but the starting formula was wrong.",
                    "improve": "Substituting back confirms the error: 3^2 + 5(3) + 6 = 30 ≠ 0. With the correct formula: (-2)^2 + 5(-2) + 6 = 0 ✓ and (-3)^2 + 5(-3) + 6 = 0 ✓.",
                },
            ],
        },
        # ── Fully correct: factorisation ──────────────────────────────────────
        {
            "question": "Solve x^2 - 5x + 6 = 0",
            "method": "factorisation",
            "assigned_marks": 4.0,
            "total_marks": 4.0,
            "steps": [
                {
                    "step_number": 1,
                    "expression": "x^2 - 5x + 6 = 0",
                    "validity": "correct",
                    "marks_awarded": 1.0,
                    "error_description": None,
                },
                {
                    "step_number": 2,
                    "expression": "(x - 2)(x - 3) = 0",
                    "validity": "correct",
                    "marks_awarded": 2.0,
                    "error_description": None,
                },
                {
                    "step_number": 3,
                    "expression": "x = 2 or x = 3",
                    "validity": "correct",
                    "marks_awarded": 1.0,
                    "error_description": None,
                },
            ],
            "marking_scheme": [
                {"step_number": 1, "expected_expression": "x^2 - 5x + 6 = 0", "marks": 1.0, "description": "State the equation"},
                {"step_number": 2, "expected_expression": "(x - 2)(x - 3) = 0", "marks": 2.0, "description": "Correct factorisation"},
                {"step_number": 3, "expected_expression": "x = 2 or x = 3", "marks": 1.0, "description": "Both roots correct"},
            ],
            "step_feedback": [
                {
                    "step_number": 1,
                    "correct": "You correctly identified the quadratic equation.",
                    "missing": None,
                    "deduction": None,
                    "improve": "Well done, continue to the next step.",
                },
                {
                    "step_number": 2,
                    "correct": "Excellent factorisation — (x - 2)(x - 3) is correct. You found the factor pair -2 and -3 that multiplies to +6 and adds to -5.",
                    "missing": None,
                    "deduction": None,
                    "improve": "Well done, continue to the next step.",
                },
                {
                    "step_number": 3,
                    "correct": "Both roots x = 2 and x = 3 are correct.",
                    "missing": None,
                    "deduction": None,
                    "improve": "Full marks. As good practice, verify by substituting: 2^2 - 5(2) + 6 = 0 ✓ and 3^2 - 5(3) + 6 = 0 ✓.",
                },
            ],
        },
        # ── Fully correct: quadratic formula ──────────────────────────────────
        {
            "question": "Solve x^2 - 3x - 10 = 0",
            "method": "quadratic formula",
            "assigned_marks": 5.0,
            "total_marks": 5.0,
            "steps": [
                {
                    "step_number": 1,
                    "expression": "x = (3 ± √(9 + 40)) / 2",
                    "validity": "correct",
                    "marks_awarded": 2.0,
                    "error_description": None,
                },
                {
                    "step_number": 2,
                    "expression": "x = (3 ± √49) / 2 = (3 ± 7) / 2",
                    "validity": "correct",
                    "marks_awarded": 2.0,
                    "error_description": None,
                },
                {
                    "step_number": 3,
                    "expression": "x = 5 or x = -2",
                    "validity": "correct",
                    "marks_awarded": 1.0,
                    "error_description": None,
                },
            ],
            "marking_scheme": [
                {"step_number": 1, "expected_expression": "x = (3 ± √(9 + 40)) / 2", "marks": 2.0, "description": "Apply quadratic formula with correct a, b, c values"},
                {"step_number": 2, "expected_expression": "x = (3 ± 7) / 2", "marks": 2.0, "description": "Correctly evaluate the discriminant"},
                {"step_number": 3, "expected_expression": "x = 5 or x = -2", "marks": 1.0, "description": "Both roots correct"},
            ],
            "step_feedback": [
                {
                    "step_number": 1,
                    "correct": "You correctly applied the quadratic formula with a = 1, b = -3, c = -10, giving x = (3 ± √(9 + 40)) / 2.",
                    "missing": None,
                    "deduction": None,
                    "improve": "Well done, continue to the next step.",
                },
                {
                    "step_number": 2,
                    "correct": "The discriminant is computed correctly: 9 + 40 = 49, and √49 = 7.",
                    "missing": None,
                    "deduction": None,
                    "improve": "Well done, continue to the next step.",
                },
                {
                    "step_number": 3,
                    "correct": "Both roots are correct: (3 + 7)/2 = 5 and (3 - 7)/2 = -2.",
                    "missing": None,
                    "deduction": None,
                    "improve": "Full marks — excellent use of the quadratic formula.",
                },
            ],
        },
    ]

    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(samples, f, indent=2, ensure_ascii=False)

    print(f"Sample annotations created ({len(samples)} examples) → {output_file}")


if __name__ == "__main__":
    create_sample_annotations("app/training/data/raw_annotations.json")
    prepare_dataset(
        input_file="app/training/data/raw_annotations.json",
        output_file="app/training/data/feedback_dataset.json",
    )
