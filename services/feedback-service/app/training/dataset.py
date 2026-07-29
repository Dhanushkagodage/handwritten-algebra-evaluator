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
import random
from typing import Dict, List, Optional, Tuple

from app.shared_format import FORMAT_INSTRUCTION as _FORMAT_INSTRUCTION

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


def _write_json(data: List[Dict], output_file: str) -> None:
    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"Prepared {len(data)} examples → {output_file}")


def split_dataset(
    examples: List[Dict], eval_ratio: float = 0.15, seed: int = 42
) -> Tuple[List[Dict], List[Dict]]:
    """Deterministic shuffle + split so the eval set is stable across regenerations."""
    rng = random.Random(seed)
    shuffled = examples[:]
    rng.shuffle(shuffled)
    n_eval = max(1, round(len(shuffled) * eval_ratio))
    return shuffled[n_eval:], shuffled[:n_eval]


# ─────────────────────────────────────────────────────────────────────────────
# Raw-annotation builder helpers
# ─────────────────────────────────────────────────────────────────────────────

def _step(
    n: int, expr: str, validity: str, marks: float, error: Optional[str] = None
) -> Dict:
    return {
        "step_number": n,
        "expression": expr,
        "validity": validity,
        "marks_awarded": marks,
        "error_description": error,
    }


def _scheme(n: int, expr: str, marks: float, desc: Optional[str] = None) -> Dict:
    return {
        "step_number": n,
        "expected_expression": expr,
        "marks": marks,
        "description": desc,
    }


def _fb(
    n: int,
    correct: str,
    missing: Optional[str] = None,
    deduction: Optional[str] = None,
    improve: str = "Well done, continue to the next step.",
) -> Dict:
    return {
        "step_number": n,
        "correct": correct,
        "missing": missing,
        "deduction": deduction,
        "improve": improve,
    }


def _example(
    question: str,
    method: str,
    assigned: float,
    total: float,
    steps: List[Dict],
    scheme: List[Dict],
    feedback: List[Dict],
) -> Dict:
    return {
        "question": question,
        "method": method,
        "assigned_marks": assigned,
        "total_marks": total,
        "steps": steps,
        "marking_scheme": scheme,
        "step_feedback": feedback,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Topic: Linear equations
# ─────────────────────────────────────────────────────────────────────────────

def _linear_examples() -> List[Dict]:
    return [
        # Fully correct
        _example(
            "Solve 3x + 5 = 20", "linear equation", 2.0, 2.0,
            steps=[
                _step(1, "3x = 15", "correct", 1.0),
                _step(2, "x = 5", "correct", 1.0),
            ],
            scheme=[
                _scheme(1, "3x = 15", 1.0, "Isolate the x term"),
                _scheme(2, "x = 5", 1.0, "Solve for x"),
            ],
            feedback=[
                _fb(1, "You correctly subtracted 5 from both sides."),
                _fb(2, "Dividing both sides by 3 gives the correct answer.",
                    improve="Full marks. Verify: 3(5) + 5 = 20 ✓."),
            ],
        ),
        # Sign error moving a term across the equals sign
        _example(
            "Solve 2x - 7 = 3", "linear equation", 0.5, 2.0,
            steps=[
                _step(1, "2x - 7 = 3", "correct", 0.5),
                _step(2, "2x = 3 - 7 = -4", "incorrect", 0.0,
                      "Sign error: moving -7 to the other side should give +7"),
                _step(3, "x = -2", "incorrect", 0.0,
                      "Follows from the sign error in step 2"),
            ],
            scheme=[
                _scheme(1, "2x - 7 = 3", 0.5, "Restate the equation"),
                _scheme(2, "2x = 10", 1.0, "Add 7 to both sides"),
                _scheme(3, "x = 5", 0.5, "Divide both sides by 2"),
            ],
            feedback=[
                _fb(1, "You correctly restated the equation."),
                _fb(2, "You attempted to isolate the x term.",
                    missing="When you move -7 to the other side, it becomes +7, so 2x = 3 + 7 = 10, not 3 - 7 = -4.",
                    deduction="Full marks lost because the sign was not flipped when moving the term across the equals sign."),
                _fb(3, "You correctly divided your (incorrect) 2x value by 2.",
                    missing="Since step 2 was wrong, x = -2 is wrong. From 2x = 10, x = 5.",
                    deduction="Marks lost because this follows directly from the sign error above.",
                    improve="Always double the check: when a term crosses the equals sign, its sign flips. Verify: 2(5) - 7 = 3 ✓."),
            ],
        ),
        # Incomplete simplification (unsimplified fraction)
        _example(
            "Solve 4x - 12 = 0", "linear equation", 1.5, 2.0,
            steps=[
                _step(1, "4x = 12", "correct", 1.0),
                _step(2, "x = 12/4", "partial", 0.5,
                      "Answer left as unsimplified fraction — must evaluate to x = 3"),
            ],
            scheme=[
                _scheme(1, "4x = 12", 1.0, "Isolate the x term"),
                _scheme(2, "x = 3", 1.0, "Solve for x (simplified)"),
            ],
            feedback=[
                _fb(1, "You correctly added 12 to isolate the x term."),
                _fb(2, "Dividing both sides by 4 is the correct operation.",
                    missing="The fraction 12/4 must be evaluated to a final numeric value.",
                    deduction="Half a mark deducted because the answer was left as an unsimplified fraction.",
                    improve="Always compute the division: 12 ÷ 4 = 3, so write x = 3 as your final answer."),
            ],
        ),
        # Incorrect method — bracket distribution error
        _example(
            "Solve 2(x - 3) = 10", "linear equation", 0.5, 2.5,
            steps=[
                _step(1, "2x - 3 = 10", "incorrect", 0.0,
                      "Distribution error: 2(x-3) = 2x - 6, not 2x - 3"),
                _step(2, "2x = 13", "incorrect", 0.0, "Follows from the distribution error"),
                _step(3, "x = 6.5", "incorrect", 0.5, "Method (isolate then divide) is otherwise sound"),
            ],
            scheme=[
                _scheme(1, "2x - 6 = 10", 1.0, "Distribute the 2 across the bracket"),
                _scheme(2, "2x = 16", 1.0, "Add 6 to both sides"),
                _scheme(3, "x = 8", 0.5, "Divide both sides by 2"),
            ],
            feedback=[
                _fb(1, "You correctly identified that the bracket must be expanded first.",
                    missing="Both terms inside the bracket must be multiplied by 2: 2(x - 3) = 2x - 6, not 2x - 3.",
                    deduction="Full marks lost because the 3 was not multiplied by 2 when distributing."),
                _fb(2, "You correctly added the constant to isolate the x term (given your Step 1).",
                    missing="Because Step 1 was wrong, 2x = 13 is wrong. The correct equation is 2x = 16.",
                    deduction="Marks lost because this follows from the distribution error."),
                _fb(3, "Your method of dividing by 2 to solve for x is correct.",
                    missing="Because of the earlier error, x = 6.5 is wrong. The correct answer is x = 8.",
                    deduction="Half credit given for correct final-step method despite the wrong number.",
                    improve="When expanding a(b - c), multiply BOTH b and c by a. Verify: 2(8 - 3) = 10 ✓."),
            ],
        ),
        # Correct method, arithmetic slip
        _example(
            "Solve 4x + 9 = 33", "linear equation", 1.0, 2.0,
            steps=[
                _step(1, "4x = 24", "correct", 1.0),
                _step(2, "x = 8", "incorrect", 0.0, "Arithmetic slip: 24 ÷ 4 = 6, not 8"),
            ],
            scheme=[
                _scheme(1, "4x = 24", 1.0, "Subtract 9 from both sides"),
                _scheme(2, "x = 6", 1.0, "Divide both sides by 4"),
            ],
            feedback=[
                _fb(1, "You correctly subtracted 9 from both sides."),
                _fb(2, "You used the correct operation (dividing by 4).",
                    missing="24 ÷ 4 = 6, not 8 — this looks like a simple arithmetic slip.",
                    deduction="Marks lost for the incorrect division, even though the method was correct.",
                    improve="Double-check basic divisions, especially under exam time pressure. Verify: 4(6) + 9 = 33 ✓."),
            ],
        ),
        # Fully correct, fractional coefficient
        _example(
            "Solve (2x + 1)/3 = 5", "linear equation", 3.0, 3.0,
            steps=[
                _step(1, "2x + 1 = 15", "correct", 1.0),
                _step(2, "2x = 14", "correct", 1.0),
                _step(3, "x = 7", "correct", 1.0),
            ],
            scheme=[
                _scheme(1, "2x + 1 = 15", 1.0, "Multiply both sides by 3"),
                _scheme(2, "2x = 14", 1.0, "Subtract 1 from both sides"),
                _scheme(3, "x = 7", 1.0, "Divide both sides by 2"),
            ],
            feedback=[
                _fb(1, "You correctly cleared the fraction by multiplying both sides by 3."),
                _fb(2, "You correctly isolated the x term."),
                _fb(3, "Correct final answer.", improve="Full marks. Verify: (2(7)+1)/3 = 15/3 = 5 ✓."),
            ],
        ),
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Topic: Simultaneous equations (linear-linear and linear-quadratic)
# ─────────────────────────────────────────────────────────────────────────────

def _simultaneous_examples() -> List[Dict]:
    return [
        # Fully correct elimination
        _example(
            "Solve: 2x + y = 7, x - y = 2", "elimination", 4.0, 4.0,
            steps=[
                _step(1, "2x + y = 7 ... (1)\nx - y = 2 ... (2)", "correct", 1.0),
                _step(2, "Adding (1) and (2): 3x = 9", "correct", 1.0),
                _step(3, "x = 3", "correct", 0.5),
                _step(4, "y = 7 - 2(3) = 1", "correct", 1.5),
            ],
            scheme=[
                _scheme(1, "2x + y = 7, x - y = 2", 1.0, "State both equations"),
                _scheme(2, "3x = 9", 1.0, "Add the equations to eliminate y"),
                _scheme(3, "x = 3", 0.5, "Solve for x"),
                _scheme(4, "y = 1", 1.5, "Substitute back to find y"),
            ],
            feedback=[
                _fb(1, "You correctly stated both equations."),
                _fb(2, "Adding the equations correctly eliminates y since +y and -y cancel."),
                _fb(3, "Correct division to solve for x."),
                _fb(4, "Correct substitution back into equation (1) to find y.",
                    improve="Full marks. Verify: 2(3)+1=7 ✓ and 3-1=2 ✓."),
            ],
        ),
        # Sign error substituting back
        _example(
            "Solve: x + 2y = 8, x - y = 2", "elimination", 3.0, 4.0,
            steps=[
                _step(1, "x + 2y = 8 ... (1)\nx - y = 2 ... (2)", "correct", 1.0),
                _step(2, "Subtracting (2) from (1): 3y = 6 → y = 2", "correct", 1.5),
                _step(3, "Substitute into (2): x - 2 = 2", "correct", 0.5),
                _step(4, "x = 2 - 2 = 0", "incorrect", 0.0,
                      "Sign error: x - 2 = 2 means x = 2 + 2 = 4, not 2 - 2"),
            ],
            scheme=[
                _scheme(1, "x + 2y = 8, x - y = 2", 1.0, "State both equations"),
                _scheme(2, "y = 2", 1.5, "Subtract to eliminate x"),
                _scheme(3, "x - 2 = 2", 0.5, "Substitute y back in"),
                _scheme(4, "x = 4", 1.0, "Solve for x"),
            ],
            feedback=[
                _fb(1, "You correctly stated both equations."),
                _fb(2, "Correct elimination — subtracting removes x, giving 3y = 6, so y = 2."),
                _fb(3, "Correct substitution of y = 2 into equation (2)."),
                _fb(4, "You attempted to isolate x by moving -2 across the equals sign.",
                    missing="x - 2 = 2 means x = 2 + 2 = 4, not 2 - 2 = 0.",
                    deduction="Marks lost for a sign error when moving -2 to the other side.",
                    improve="When solving x - a = b, add a to both sides: x = b + a. Verify: 4 - 2 = 2 ✓ and 4 + 2(2) = 8 ✓."),
            ],
        ),
        # Linear-quadratic — missing second solution
        _example(
            "Solve: y = x + 1, y = x^2 - 5", "substitution (linear-quadratic)", 4.0, 5.0,
            steps=[
                _step(1, "x + 1 = x^2 - 5", "correct", 1.0),
                _step(2, "x^2 - x - 6 = 0", "correct", 1.0),
                _step(3, "(x - 3)(x + 2) = 0", "correct", 1.5),
                _step(4, "x = 3, y = 4", "partial", 0.5,
                      "Missing second solution x = -2, y = -1"),
            ],
            scheme=[
                _scheme(1, "x + 1 = x^2 - 5", 1.0, "Substitute to eliminate y"),
                _scheme(2, "x^2 - x - 6 = 0", 1.0, "Rearrange to standard form"),
                _scheme(3, "(x - 3)(x + 2) = 0", 1.5, "Factorise"),
                _scheme(4, "(3, 4) and (-2, -1)", 1.5, "Both solution pairs"),
            ],
            feedback=[
                _fb(1, "Correct substitution of y = x + 1 into the quadratic."),
                _fb(2, "Correctly rearranged into standard quadratic form."),
                _fb(3, "Correct factorisation — factor pair 3 and -2 multiplies to -6 and sums to -1."),
                _fb(4, "x = 3, y = 4 is one correct solution pair.",
                    missing="A quadratic gives two x values. The second root x = -2 gives y = -2 + 1 = -1.",
                    deduction="1 mark lost because only one of the two solution pairs was given.",
                    improve="After factorising a quadratic from simultaneous equations, always find BOTH x values and their matching y values. Full solution: (3, 4) and (-2, -1)."),
            ],
        ),
        # Linear-quadratic — incorrect factorisation
        _example(
            "Solve: y = x + 4, y = x^2 - 2", "substitution (linear-quadratic)", 2.0, 5.0,
            steps=[
                _step(1, "x + 4 = x^2 - 2", "correct", 1.0),
                _step(2, "x^2 - x - 6 = 0", "correct", 1.0),
                _step(3, "(x - 2)(x + 3) = 0", "incorrect", 0.0,
                      "Wrong factors: (-2)(3) = -6 but -2+3 = 1, not -1"),
                _step(4, "x = 2 or x = -3", "incorrect", 0.0, "Follows from the incorrect factorisation"),
            ],
            scheme=[
                _scheme(1, "x + 4 = x^2 - 2", 1.0, "Substitute to eliminate y"),
                _scheme(2, "x^2 - x - 6 = 0", 1.0, "Rearrange to standard form"),
                _scheme(3, "(x - 3)(x + 2) = 0", 1.5, "Factorise"),
                _scheme(4, "(3, 7) and (-2, 2)", 1.5, "Both solution pairs"),
            ],
            feedback=[
                _fb(1, "Correct substitution."),
                _fb(2, "Correctly rearranged into standard quadratic form."),
                _fb(3, "You correctly set up the factorisation approach.",
                    missing="You need factors of -6 that sum to -1: those are -3 and 2, giving (x - 3)(x + 2), not (x - 2)(x + 3).",
                    deduction="Full marks lost — (x-2)(x+3) expands to x^2+x-6, which does not match x^2-x-6."),
                _fb(4, "You correctly applied the zero-product property to your factors.",
                    missing="Because Step 3 was factorised incorrectly, x = 2 and x = -3 are wrong. The correct roots are x = 3 and x = -2, giving points (3, 7) and (-2, 2).",
                    deduction="Marks lost because the roots follow from an incorrect factorisation.",
                    improve="Always verify a factorisation by expanding it back out before using it. Check: (x-3)(x+2) = x^2-x-6 ✓."),
            ],
        ),
        # Fully correct substitution
        _example(
            "Solve: y = 3x, 2x + y = 15", "substitution", 3.0, 3.0,
            steps=[
                _step(1, "2x + 3x = 15", "correct", 1.0),
                _step(2, "5x = 15 → x = 3", "correct", 1.0),
                _step(3, "y = 3(3) = 9", "correct", 1.0),
            ],
            scheme=[
                _scheme(1, "2x + 3x = 15", 1.0, "Substitute y = 3x"),
                _scheme(2, "x = 3", 1.0, "Solve for x"),
                _scheme(3, "y = 9", 1.0, "Substitute back for y"),
            ],
            feedback=[
                _fb(1, "Correct substitution of y = 3x into the second equation."),
                _fb(2, "Correctly combined like terms and solved for x."),
                _fb(3, "Correct substitution to find y.", improve="Full marks. Verify: 2(3)+9=15 ✓."),
            ],
        ),
        # Arithmetic slip
        _example(
            "Solve: x + y = 10, x - y = 4", "elimination", 2.0, 4.0,
            steps=[
                _step(1, "x + y = 10 ... (1)\nx - y = 4 ... (2)", "correct", 1.0),
                _step(2, "Adding: 2x = 14", "correct", 1.0),
                _step(3, "x = 8", "incorrect", 0.0, "Arithmetic slip: 14 ÷ 2 = 7, not 8"),
                _step(4, "y = 10 - 8 = 2", "incorrect", 0.0, "Follows from the arithmetic slip"),
            ],
            scheme=[
                _scheme(1, "x + y = 10, x - y = 4", 1.0, "State both equations"),
                _scheme(2, "2x = 14", 1.0, "Add to eliminate y"),
                _scheme(3, "x = 7", 1.0, "Solve for x"),
                _scheme(4, "y = 3", 1.0, "Substitute back for y"),
            ],
            feedback=[
                _fb(1, "You correctly stated both equations."),
                _fb(2, "Adding correctly eliminates y."),
                _fb(3, "Correct method (dividing by 2).",
                    missing="14 ÷ 2 = 7, not 8 — a simple division slip.",
                    deduction="Marks lost for the incorrect division despite the correct method."),
                _fb(4, "Correct method of substituting back.",
                    missing="Because x = 8 was wrong, y = 2 is wrong. The correct value is y = 3.",
                    deduction="Marks lost because this follows from the earlier arithmetic slip.",
                    improve="Double-check basic arithmetic. Verify: 7+3=10 ✓ and 7-3=4 ✓."),
            ],
        ),
        # Fully correct linear-quadratic (circle-style), handles ± correctly
        _example(
            "Solve: y = x, x^2 + y^2 = 8", "substitution (linear-quadratic)", 4.0, 4.0,
            steps=[
                _step(1, "x^2 + x^2 = 8 → 2x^2 = 8", "correct", 1.0),
                _step(2, "x^2 = 4", "correct", 1.0),
                _step(3, "x = ±2", "correct", 1.0),
                _step(4, "(x, y) = (2, 2) or (-2, -2)", "correct", 1.0),
            ],
            scheme=[
                _scheme(1, "2x^2 = 8", 1.0, "Substitute y = x"),
                _scheme(2, "x^2 = 4", 1.0, "Divide both sides by 2"),
                _scheme(3, "x = ±2", 1.0, "Take the square root"),
                _scheme(4, "(2,2) and (-2,-2)", 1.0, "Both solution pairs"),
            ],
            feedback=[
                _fb(1, "Correct substitution of y = x."),
                _fb(2, "Correctly divided both sides by 2."),
                _fb(3, "Correctly took both the positive and negative square roots."),
                _fb(4, "Both solution pairs correctly stated.",
                    improve="Full marks. Verify: 2^2+2^2=8 ✓ and (-2)^2+(-2)^2=8 ✓."),
            ],
        ),
        # Incomplete simplification (repeated root)
        _example(
            "Solve: y = 4x - 4, y = x^2", "substitution (linear-quadratic)", 4.0, 5.0,
            steps=[
                _step(1, "4x - 4 = x^2", "correct", 1.0),
                _step(2, "x^2 - 4x + 4 = 0", "correct", 1.0),
                _step(3, "(x - 2)^2 = 0", "correct", 1.5),
                _step(4, "x = 4/2", "partial", 0.5,
                      "Not simplified — should state x = 2 (repeated root) and y = 4"),
            ],
            scheme=[
                _scheme(1, "4x - 4 = x^2", 1.0, "Substitute to eliminate y"),
                _scheme(2, "x^2 - 4x + 4 = 0", 1.0, "Rearrange to standard form"),
                _scheme(3, "(x - 2)^2 = 0", 1.5, "Factorise"),
                _scheme(4, "x = 2, y = 4", 1.5, "State the repeated root and matching y"),
            ],
            feedback=[
                _fb(1, "Correct substitution."),
                _fb(2, "Correctly rearranged into standard quadratic form."),
                _fb(3, "Correct factorisation as a perfect square."),
                _fb(4, "You correctly began solving x - 2 = 0.",
                    missing="The expression 4/2 must be evaluated: x = 2. Then y = 4(2) - 4 = 4.",
                    deduction="Half a mark lost because the final numeric values were not simplified/stated.",
                    improve="Always finish arithmetic to a final number, and state both x and y clearly at the end."),
            ],
        ),
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Topic: Quadratic equations (factorisation, difference of squares, formula,
# completing the square)
# ─────────────────────────────────────────────────────────────────────────────

def _quadratic_examples() -> List[Dict]:
    return [
        # Sign error in factorisation result
        _example(
            "Solve x^2 + x - 6 = 0", "factorisation", 3.5, 4.0,
            steps=[
                _step(1, "x^2 + x - 6 = 0", "correct", 1.0),
                _step(2, "(x + 3)(x - 2) = 0", "correct", 2.0),
                _step(3, "x = -3 or x = -2", "incorrect", 0.5,
                      "Sign error: (x - 2) = 0 gives x = +2, not x = -2"),
            ],
            scheme=[
                _scheme(1, "x^2 + x - 6 = 0", 1.0, "State the equation"),
                _scheme(2, "(x + 3)(x - 2) = 0", 2.0, "Correct factorisation"),
                _scheme(3, "x = -3 or x = 2", 1.0, "Both roots correct"),
            ],
            feedback=[
                _fb(1, "You correctly identified and stated the quadratic equation."),
                _fb(2, "Excellent — your factorisation (x + 3)(x - 2) = 0 is completely correct."),
                _fb(3, "You applied the zero-product property and correctly found x = -3 from (x + 3) = 0.",
                    missing="The second root is wrong. From (x - 2) = 0, adding 2 to both sides gives x = +2, not x = -2.",
                    deduction="Half a mark awarded for method; half lost because x = -2 has the wrong sign — the correct second root is x = +2.",
                    improve="When solving (x - a) = 0, the root is x = +a, not -a. Always verify by substituting back: (-2)^2 + (-2) - 6 = -4 ≠ 0, confirming x = -2 is wrong."),
            ],
        ),
        # Missing second root (difference of squares)
        _example(
            "Solve x^2 - 9 = 0", "difference of squares", 2.0, 3.0,
            steps=[
                _step(1, "(x - 3)(x + 3) = 0", "correct", 1.5),
                _step(2, "x = 3", "partial", 0.5, "Missing second root x = -3"),
            ],
            scheme=[
                _scheme(1, "(x - 3)(x + 3) = 0", 1.5, "Factor using difference of squares"),
                _scheme(2, "x = 3 or x = -3", 1.5, "State both roots"),
            ],
            feedback=[
                _fb(1, "Perfect application of the difference-of-squares identity: a^2 - b^2 = (a - b)(a + b)."),
                _fb(2, "x = 3 is one correct root, obtained correctly from (x - 3) = 0.",
                    missing="The second root x = -3 is missing. From (x + 3) = 0, subtracting 3 from both sides gives x = -3.",
                    deduction="1 mark deducted because a factored quadratic has two roots — only one was stated.",
                    improve="After applying the zero-product property, solve each factor separately: (x - 3) = 0 → x = 3 and (x + 3) = 0 → x = -3."),
            ],
        ),
        # Missing negative root (direct manipulation)
        _example(
            "Solve 2x^2 - 8 = 0", "direct algebraic manipulation", 2.5, 4.0,
            steps=[
                _step(1, "2x^2 = 8", "correct", 1.0),
                _step(2, "x^2 = 4", "correct", 1.0),
                _step(3, "x = 2", "partial", 0.5, "Missing negative root x = -2; x^2 = 4 has two solutions"),
            ],
            scheme=[
                _scheme(1, "2x^2 = 8", 1.0, "Rearrange to isolate x^2 term"),
                _scheme(2, "x^2 = 4", 1.0, "Divide both sides by 2"),
                _scheme(3, "x = ±2", 2.0, "Both positive and negative roots"),
            ],
            feedback=[
                _fb(1, "You correctly rearranged the equation by adding 8 to both sides."),
                _fb(2, "Dividing both sides by 2 to get x^2 = 4 is correct."),
                _fb(3, "x = 2 is one valid solution — 2^2 = 4 ✓.",
                    missing="The equation x^2 = 4 also has the solution x = -2, since (-2)^2 = 4. The complete answer is x = ±2.",
                    deduction="1.5 marks deducted for the missing negative root — square roots always yield two solutions unless the context restricts to positive values.",
                    improve="When solving x^2 = k (k > 0), always write x = ±√k to capture both roots."),
            ],
        ),
        # Incorrect factorisation
        _example(
            "Solve x^2 - 7x + 12 = 0", "factorisation", 1.0, 4.0,
            steps=[
                _step(1, "x^2 - 7x + 12 = 0", "correct", 1.0),
                _step(2, "(x - 4)(x + 3) = 0", "incorrect", 0.0,
                      "Wrong sign in second factor: expands to x^2 - x - 12, not x^2 - 7x + 12"),
                _step(3, "x = 4 or x = -3", "incorrect", 0.0, "Roots follow from the incorrect factorisation"),
            ],
            scheme=[
                _scheme(1, "x^2 - 7x + 12 = 0", 1.0, "State the equation"),
                _scheme(2, "(x - 4)(x - 3) = 0", 2.0, "Correct factorisation"),
                _scheme(3, "x = 4 or x = 3", 1.0, "Both roots correct"),
            ],
            feedback=[
                _fb(1, "You correctly stated the quadratic equation to be solved."),
                _fb(2, "You identified the correct factorisation approach and found one correct factor (x - 4).",
                    missing="The second factor is wrong. You need two numbers that multiply to +12 AND add to -7. Those are -4 and -3, giving (x - 4)(x - 3).",
                    deduction="2 marks deducted because (x - 4)(x + 3) = x^2 - x - 12 ≠ x^2 - 7x + 12."),
                _fb(3, "You correctly applied the zero-product property to derive roots from your factors.",
                    missing="Because Step 2 was factorised incorrectly, the roots x = 4 and x = -3 are wrong. The correct roots are x = 4 and x = 3.",
                    deduction="1 mark deducted because the roots are based on an incorrect factorisation.",
                    improve="To factorise x^2 + bx + c, list all factor pairs of c that sum to b. Here: (-4)×(-3)=12 and (-4)+(-3)=-7 ✓."),
            ],
        ),
        # Correct method, arithmetic slip
        _example(
            "Solve x^2 - 9x + 20 = 0", "factorisation", 3.0, 4.0,
            steps=[
                _step(1, "x^2 - 9x + 20 = 0", "correct", 1.0),
                _step(2, "(x - 4)(x - 5) = 0", "correct", 2.0),
                _step(3, "x = 4 or x = 6", "incorrect", 0.0,
                      "Arithmetic slip: (x - 5) = 0 gives x = 5, not x = 6"),
            ],
            scheme=[
                _scheme(1, "x^2 - 9x + 20 = 0", 1.0, "State the equation"),
                _scheme(2, "(x - 4)(x - 5) = 0", 2.0, "Correct factorisation"),
                _scheme(3, "x = 4 or x = 5", 1.0, "Both roots correct"),
            ],
            feedback=[
                _fb(1, "You correctly stated the quadratic equation."),
                _fb(2, "Excellent factorisation — (x - 4)(x - 5) is completely correct."),
                _fb(3, "x = 4 is correct, obtained from (x - 4) = 0.",
                    missing="The second root is wrong. From (x - 5) = 0, adding 5 to both sides gives x = 5, not x = 6.",
                    deduction="1 mark deducted because x = 6 is incorrect — the correct second root is x = 5.",
                    improve="When reading a root from a factor (x - a) = 0, the answer is x = a. A quick check catches the error: 6^2-9(6)+20=2≠0."),
            ],
        ),
        # Correct method, wrong sign in quadratic formula
        _example(
            "Solve x^2 + 5x + 6 = 0 using the quadratic formula", "quadratic formula", 2.5, 5.0,
            steps=[
                _step(1, "x = (5 ± √(25 - 24)) / 2", "incorrect", 0.5,
                      "Sign error: formula requires -b; here b = +5, so -b = -5"),
                _step(2, "x = (5 ± 1) / 2", "correct", 1.0),
                _step(3, "x = 3 or x = 2", "incorrect", 1.0,
                      "Roots are wrong because Step 1 used +5 instead of -5"),
            ],
            scheme=[
                _scheme(1, "x = (-5 ± √(25 - 24)) / 2", 2.0, "Apply quadratic formula with -b"),
                _scheme(2, "x = (-5 ± 1) / 2", 1.0, "Simplify discriminant"),
                _scheme(3, "x = -2 or x = -3", 2.0, "Both roots correct"),
            ],
            feedback=[
                _fb(1, "You correctly identified the discriminant as 25 - 24 = 1, and the denominator as 2.",
                    missing="The numerator uses the wrong sign. The formula is x = (-b ± √(b^2-4ac))/2a. Here b=+5, so -b=-5, not +5.",
                    deduction="1.5 marks deducted because using +b instead of -b produces entirely wrong roots."),
                _fb(2, "The simplification of the square root is correct: √1 = 1, giving (5 ± 1)/2 — consistent with your Step 1."),
                _fb(3, "You correctly evaluated the two ± cases from your Step 2 expression.",
                    missing="Because Step 1 had the wrong sign, x = 3 and x = 2 are incorrect. The correct roots are x = -2 and x = -3.",
                    deduction="1 mark deducted because the roots follow from the sign error in Step 1.",
                    improve="Memorise the formula with the negative sign before b: x = (−b ± √(b²−4ac)) / 2a."),
            ],
        ),
        # Fully correct factorisation
        _example(
            "Solve x^2 - 5x + 6 = 0", "factorisation", 4.0, 4.0,
            steps=[
                _step(1, "x^2 - 5x + 6 = 0", "correct", 1.0),
                _step(2, "(x - 2)(x - 3) = 0", "correct", 2.0),
                _step(3, "x = 2 or x = 3", "correct", 1.0),
            ],
            scheme=[
                _scheme(1, "x^2 - 5x + 6 = 0", 1.0, "State the equation"),
                _scheme(2, "(x - 2)(x - 3) = 0", 2.0, "Correct factorisation"),
                _scheme(3, "x = 2 or x = 3", 1.0, "Both roots correct"),
            ],
            feedback=[
                _fb(1, "You correctly identified the quadratic equation."),
                _fb(2, "Excellent factorisation — (x - 2)(x - 3) is correct."),
                _fb(3, "Both roots x = 2 and x = 3 are correct.",
                    improve="Full marks. Verify: 2^2-5(2)+6=0 ✓ and 3^2-5(3)+6=0 ✓."),
            ],
        ),
        # Fully correct quadratic formula
        _example(
            "Solve x^2 - 3x - 10 = 0", "quadratic formula", 5.0, 5.0,
            steps=[
                _step(1, "x = (3 ± √(9 + 40)) / 2", "correct", 2.0),
                _step(2, "x = (3 ± √49) / 2 = (3 ± 7) / 2", "correct", 2.0),
                _step(3, "x = 5 or x = -2", "correct", 1.0),
            ],
            scheme=[
                _scheme(1, "x = (3 ± √(9 + 40)) / 2", 2.0, "Apply quadratic formula with correct a, b, c"),
                _scheme(2, "x = (3 ± 7) / 2", 2.0, "Correctly evaluate the discriminant"),
                _scheme(3, "x = 5 or x = -2", 1.0, "Both roots correct"),
            ],
            feedback=[
                _fb(1, "You correctly applied the quadratic formula with a=1, b=-3, c=-10."),
                _fb(2, "The discriminant is computed correctly: 9 + 40 = 49, and √49 = 7."),
                _fb(3, "Both roots are correct: (3+7)/2=5 and (3-7)/2=-2.", improve="Full marks — excellent use of the quadratic formula."),
            ],
        ),
        # Completing the square — fully correct
        _example(
            "Solve x^2 + 6x + 5 = 0 by completing the square", "completing the square", 5.0, 5.0,
            steps=[
                _step(1, "(x + 3)^2 - 9 + 5 = 0", "correct", 1.5),
                _step(2, "(x + 3)^2 = 4", "correct", 1.0),
                _step(3, "x + 3 = ±2", "correct", 1.0),
                _step(4, "x = -1 or x = -5", "correct", 1.5),
            ],
            scheme=[
                _scheme(1, "(x + 3)^2 - 9 + 5 = 0", 1.5, "Complete the square (half of 6 is 3)"),
                _scheme(2, "(x + 3)^2 = 4", 1.0, "Simplify constants"),
                _scheme(3, "x + 3 = ±2", 1.0, "Take the square root of both sides"),
                _scheme(4, "x = -1 or x = -5", 1.5, "Both roots correct"),
            ],
            feedback=[
                _fb(1, "Correctly completed the square using half of the x-coefficient (6/2=3)."),
                _fb(2, "Correctly simplified the constants."),
                _fb(3, "Correctly took both the positive and negative square roots."),
                _fb(4, "Both roots correctly stated.", improve="Full marks. Verify: (-1)^2+6(-1)+5=0 ✓ and (-5)^2+6(-5)+5=0 ✓."),
            ],
        ),
        # Completing the square — missing negative root
        _example(
            "Solve x^2 - 4x - 5 = 0 by completing the square", "completing the square", 3.0, 4.0,
            steps=[
                _step(1, "x^2 - 4x - 5 = 0 → (x - 2)^2 - 4 - 5 = 0", "correct", 1.0),
                _step(2, "(x - 2)^2 = 9", "correct", 1.0),
                _step(3, "x - 2 = 3 → x = 5", "partial", 1.0,
                      "Missing negative case: x - 2 = -3 also valid, giving x = -1"),
            ],
            scheme=[
                _scheme(1, "(x - 2)^2 - 9 = 0", 1.0, "Complete the square (half of -4 is -2)"),
                _scheme(2, "(x - 2)^2 = 9", 1.0, "Simplify constants"),
                _scheme(3, "x = 5 or x = -1", 2.0, "Both roots correct"),
            ],
            feedback=[
                _fb(1, "Correctly completed the square using half of the x-coefficient (-4/2=-2)."),
                _fb(2, "Correctly simplified the constants."),
                _fb(3, "x = 5 is one correct root, obtained from x - 2 = 3.",
                    missing="Taking the square root gives x - 2 = ±3, not just +3. The negative case gives x - 2 = -3, so x = -1.",
                    deduction="1 mark deducted for the missing second root — completing the square always requires the ± sign.",
                    improve="When you reach (x-a)^2 = k, always write x - a = ±√k to capture both solutions."),
            ],
        ),
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Topic: Indices and surds
# ─────────────────────────────────────────────────────────────────────────────

def _indices_surds_examples() -> List[Dict]:
    return [
        # Fully correct index laws
        _example(
            "Simplify (x^3)(x^5) ÷ x^2", "index laws", 2.0, 2.0,
            steps=[
                _step(1, "x^3 × x^5 = x^8", "correct", 1.0),
                _step(2, "x^8 ÷ x^2 = x^6", "correct", 1.0),
            ],
            scheme=[
                _scheme(1, "x^8", 1.0, "Add exponents when multiplying"),
                _scheme(2, "x^6", 1.0, "Subtract exponents when dividing"),
            ],
            feedback=[
                _fb(1, "Correctly added the exponents: 3 + 5 = 8."),
                _fb(2, "Correctly subtracted the exponents: 8 - 2 = 6.", improve="Full marks — correct use of index laws."),
            ],
        ),
        # Sign error with negative index
        _example(
            "Simplify x^-2 × x^5", "index laws", 0.0, 2.0,
            steps=[
                _step(1, "x^-2 × x^5 = x^-7", "incorrect", 0.0,
                      "Sign error: exponents add, -2+5=3, not -2-5=-7"),
            ],
            scheme=[
                _scheme(1, "x^3", 2.0, "Add the exponents"),
            ],
            feedback=[
                _fb(1, "You correctly recognised that the index law for multiplication applies.",
                    missing="When multiplying powers of the same base, exponents ADD: -2 + 5 = 3, not -2 - 5 = -7.",
                    deduction="Full marks lost because the exponents were subtracted instead of added.",
                    improve="Remember: a^m × a^n = a^(m+n), always add the exponents, regardless of their sign."),
            ],
        ),
        # Incomplete — coefficient not raised to the power
        _example(
            "Simplify (2x^2)^3", "index laws", 0.5, 2.0,
            steps=[
                _step(1, "(2x^2)^3 = 2x^6", "partial", 0.5,
                      "Coefficient not cubed — 2^3 = 8, not 2"),
            ],
            scheme=[
                _scheme(1, "8x^6", 2.0, "Raise both the coefficient and x^2 to the power 3"),
            ],
            feedback=[
                _fb(1, "You correctly applied the power rule to the x^2 term, giving x^6.",
                    missing="The coefficient 2 must also be raised to the power 3: 2^3 = 8, so the full answer is 8x^6, not 2x^6.",
                    deduction="1.5 marks lost because the coefficient was left un-cubed.",
                    improve="When raising a product (ab)^n, every factor inside the brackets — including numeric coefficients — is raised to the power n."),
            ],
        ),
        # Incorrect method — fractional index treated as multiplication
        _example(
            "Simplify 8^(2/3)", "index laws (fractional indices)", 0.0, 2.0,
            steps=[
                _step(1, "8^(2/3) = 8 × 2/3 = 16/3", "incorrect", 0.0,
                      "Fractional exponents mean roots/powers, not multiplication"),
            ],
            scheme=[
                _scheme(1, "4", 2.0, "8^(2/3) = (cube root of 8)^2 = 2^2 = 4"),
            ],
            feedback=[
                _fb(1, "You attempted to evaluate the expression.",
                    missing="A fractional exponent a^(m/n) means (n-th root of a)^m, not a multiplied by m/n. Here 8^(2/3) = (∛8)^2 = 2^2 = 4.",
                    deduction="Full marks lost because the meaning of a fractional index was misunderstood.",
                    improve="Remember a^(m/n) = (ⁿ√a)^m. Break it into two steps: take the root first, then apply the power."),
            ],
        ),
        # Correct method, arithmetic slip
        _example(
            "Simplify 3^2 × 3^4", "index laws", 1.0, 2.0,
            steps=[
                _step(1, "3^2 × 3^4 = 3^6", "correct", 1.0),
                _step(2, "3^6 = 627", "incorrect", 0.0, "Arithmetic slip: 3^6 = 729, not 627"),
            ],
            scheme=[
                _scheme(1, "3^6", 1.0, "Add the exponents"),
                _scheme(2, "729", 1.0, "Evaluate 3^6"),
            ],
            feedback=[
                _fb(1, "Correctly added the exponents: 2 + 4 = 6."),
                _fb(2, "Correct method of evaluating the power.",
                    missing="3^6 = 729, not 627 — this appears to be an arithmetic slip.",
                    deduction="Marks lost for the incorrect evaluation despite the correct method.",
                    improve="Break the calculation into smaller steps: 3^6 = 3^3 × 3^3 = 27 × 27 = 729."),
            ],
        ),
        # Surds — fully correct simplification
        _example(
            "Simplify √50", "surds", 2.0, 2.0,
            steps=[
                _step(1, "√50 = √(25 × 2)", "correct", 1.0),
                _step(2, "= 5√2", "correct", 1.0),
            ],
            scheme=[
                _scheme(1, "√(25 × 2)", 1.0, "Find the largest perfect-square factor"),
                _scheme(2, "5√2", 1.0, "Simplify the surd"),
            ],
            feedback=[
                _fb(1, "Correctly identified 25 as the largest perfect-square factor of 50."),
                _fb(2, "Correctly simplified √25 to 5.", improve="Full marks — clean simplification of the surd."),
            ],
        ),
        # Surds — incomplete rationalisation
        _example(
            "Rationalise 1/√3", "surds (rationalisation)", 1.0, 2.0,
            steps=[
                _step(1, "Multiply top and bottom by √3: √3 / (√3 × √3)", "correct", 1.0),
                _step(2, "= √3", "incorrect", 0.0,
                      "Denominator √3×√3=3 was dropped — correct answer is √3/3"),
            ],
            scheme=[
                _scheme(1, "√3 / (√3 × √3)", 1.0, "Multiply top and bottom by √3"),
                _scheme(2, "√3 / 3", 1.0, "Simplify the denominator"),
            ],
            feedback=[
                _fb(1, "Correct method — multiplying by √3/√3 is the right way to rationalise."),
                _fb(2, "You correctly simplified the numerator.",
                    missing="The denominator √3 × √3 = 3 was dropped from the final answer — it should be √3 / 3, not just √3.",
                    deduction="1 mark deducted because the final answer is missing its denominator.",
                    improve="After multiplying by the conjugate/rationalising factor, always simplify BOTH the numerator and denominator, and keep both in your final answer."),
            ],
        ),
        # Surds — fully correct addition
        _example(
            "Simplify 2√3 + 5√3", "surds", 1.0, 1.0,
            steps=[
                _step(1, "2√3 + 5√3 = 7√3", "correct", 1.0),
            ],
            scheme=[
                _scheme(1, "7√3", 1.0, "Add like surds by adding their coefficients"),
            ],
            feedback=[
                _fb(1, "Correct — like surds add just like like terms: 2 + 5 = 7.", improve="Full marks."),
            ],
        ),
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Topic: Polynomials (expansion, factorisation, cubic factorisation)
# ─────────────────────────────────────────────────────────────────────────────

def _polynomial_examples() -> List[Dict]:
    return [
        # Fully correct expansion
        _example(
            "Expand (x + 2)(x + 5)", "expansion", 1.0, 1.0,
            steps=[
                _step(1, "x^2 + 5x + 2x + 10 = x^2 + 7x + 10", "correct", 1.0),
            ],
            scheme=[_scheme(1, "x^2 + 7x + 10", 1.0, "Expand and collect like terms")],
            feedback=[_fb(1, "Correctly expanded and collected like terms: 5x + 2x = 7x.", improve="Full marks.")],
        ),
        # Sign error in expansion
        _example(
            "Expand (x - 3)(x + 4)", "expansion", 0.5, 1.5,
            steps=[
                _step(1, "x^2 + x + 12", "incorrect", 0.5, "Sign error: (-3)(4) = -12, not +12"),
            ],
            scheme=[_scheme(1, "x^2 + x - 12", 1.5, "Expand and collect like terms")],
            feedback=[
                _fb(1, "You correctly expanded the x^2 and x terms (4x - 3x = x).",
                    missing="The constant term is (-3)(4) = -12, not +12.",
                    deduction="1 mark deducted for the sign error in the constant term.",
                    improve="When multiplying the last terms of each bracket, carefully track the sign: (-3)(+4) = -12."),
            ],
        ),
        # Incomplete common-factor extraction
        _example(
            "Factorise 6x^2 + 9x", "common factor", 1.0, 2.0,
            steps=[
                _step(1, "3(2x^2 + 3x)", "partial", 1.0, "x is also a common factor — should be 3x(2x + 3)"),
            ],
            scheme=[_scheme(1, "3x(2x + 3)", 2.0, "Factor out the highest common factor")],
            feedback=[
                _fb(1, "You correctly identified 3 as a common numeric factor.",
                    missing="x is also common to both terms — the highest common factor is 3x, giving 3x(2x + 3).",
                    deduction="1 mark deducted because the factorisation is incomplete — x was left inside the bracket unnecessarily.",
                    improve="Always check every variable for a common factor too, not just the numbers. Verify: 3x(2x+3) = 6x^2+9x ✓."),
            ],
        ),
        # Incorrect grouping factorisation
        _example(
            "Factorise x^2 + 5x + xy + 5y", "factorisation by grouping", 0.5, 2.0,
            steps=[
                _step(1, "(x + y)(x + 5y)", "incorrect", 0.5, "Incorrect grouping — does not expand back to the original expression"),
            ],
            scheme=[_scheme(1, "(x + 5)(x + y)", 2.0, "Group in pairs and factor each pair")],
            feedback=[
                _fb(1, "You attempted to factorise by pairing terms.",
                    missing="Grouping (x^2+5x)+(xy+5y) gives x(x+5)+y(x+5) = (x+5)(x+y), not (x+y)(x+5y).",
                    deduction="1.5 marks deducted because the stated factorisation does not expand back to the original expression.",
                    improve="Always verify a grouping factorisation by expanding your answer back out and checking it matches the original expression."),
            ],
        ),
        # Cubic factorisation — arithmetic slip
        _example(
            "Factorise x^3 - 8", "difference of cubes", 1.5, 2.0,
            steps=[
                _step(1, "(x - 2)(x^2 + 2x + 2)", "incorrect", 1.5, "Arithmetic slip: the constant term should be 4 (2^2), not 2"),
            ],
            scheme=[_scheme(1, "(x - 2)(x^2 + 2x + 4)", 2.0, "Apply the difference-of-cubes identity")],
            feedback=[
                _fb(1, "You correctly identified the difference-of-cubes pattern a^3-b^3=(a-b)(a^2+ab+b^2) and got the first two terms right.",
                    missing="The final term should be b^2 = 2^2 = 4, not 2.",
                    deduction="Half a mark deducted for the arithmetic slip in the last term of the trinomial.",
                    improve="Memorise a^3-b^3=(a-b)(a^2+ab+b^2) carefully — the last term is b squared, not b."),
            ],
        ),
        # Fully correct sum of cubes
        _example(
            "Factorise x^3 + 27", "sum of cubes", 2.0, 2.0,
            steps=[
                _step(1, "(x + 3)(x^2 - 3x + 9)", "correct", 2.0),
            ],
            scheme=[_scheme(1, "(x + 3)(x^2 - 3x + 9)", 2.0, "Apply the sum-of-cubes identity")],
            feedback=[
                _fb(1, "Correct application of a^3+b^3=(a+b)(a^2-ab+b^2) with b=3.", improve="Full marks."),
            ],
        ),
        # Incomplete factorisation
        _example(
            "Factorise x^3 - 4x", "common factor + difference of squares", 1.0, 2.0,
            steps=[
                _step(1, "x(x^2 - 4)", "partial", 1.0, "Factorisation not complete — x^2 - 4 is itself a difference of squares"),
            ],
            scheme=[_scheme(1, "x(x - 2)(x + 2)", 2.0, "Factor out x, then apply difference of squares")],
            feedback=[
                _fb(1, "You correctly factored out the common factor x.",
                    missing="x^2 - 4 is a difference of squares and factorises further into (x - 2)(x + 2).",
                    deduction="1 mark deducted because the factorisation was not taken to its fully factored form.",
                    improve="After factoring out a common term, always check whether what remains can be factorised further."),
            ],
        ),
        # Fully correct grouping
        _example(
            "Factorise 3x^2 + 6x + 4x + 8", "factorisation by grouping", 2.0, 2.0,
            steps=[
                _step(1, "3x(x + 2) + 4(x + 2)", "correct", 1.0),
                _step(2, "(x + 2)(3x + 4)", "correct", 1.0),
            ],
            scheme=[
                _scheme(1, "3x(x + 2) + 4(x + 2)", 1.0, "Group in pairs and factor each pair"),
                _scheme(2, "(x + 2)(3x + 4)", 1.0, "Factor out the common bracket"),
            ],
            feedback=[
                _fb(1, "Correctly grouped and factored each pair, revealing the common bracket (x + 2)."),
                _fb(2, "Correctly factored out the common bracket.", improve="Full marks. Verify: (x+2)(3x+4) = 3x^2+10x+8, matching the original expression."),
            ],
        ),
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Topic: Remainder theorem and factor theorem
# ─────────────────────────────────────────────────────────────────────────────

def _remainder_factor_theorem_examples() -> List[Dict]:
    return [
        # Fully correct remainder theorem
        _example(
            "Find the remainder when x^3 - 2x^2 + 3x - 5 is divided by (x - 1)", "remainder theorem", 2.0, 2.0,
            steps=[
                _step(1, "f(1) = 1 - 2 + 3 - 5 = -3", "correct", 2.0),
            ],
            scheme=[_scheme(1, "-3", 2.0, "Substitute x = 1 into the polynomial")],
            feedback=[_fb(1, "Correctly substituted x = 1 and evaluated the polynomial.", improve="Full marks — the remainder is -3.")],
        ),
        # Sign error in substitution value
        _example(
            "Find the remainder when x^3 + x^2 - 4 is divided by (x + 2)", "remainder theorem", 0.0, 2.0,
            steps=[
                _step(1, "f(2) = 8 + 4 - 4 = 8", "incorrect", 0.0, "Wrong substitution value: for divisor (x+2), substitute x = -2, not x = 2"),
            ],
            scheme=[_scheme(1, "-8", 2.0, "Substitute x = -2 into the polynomial")],
            feedback=[
                _fb(1, "You correctly set up the polynomial evaluation approach.",
                    missing="For divisor (x + 2), the remainder theorem requires substituting x = -2 (the value that makes x+2=0), not x = 2.",
                    deduction="Full marks lost because the wrong value was substituted.",
                    improve="For divisor (x - a), always substitute x = a. Here the divisor is (x - (-2)), so substitute x = -2."),
            ],
        ),
        # Incomplete conclusion for factor theorem
        _example(
            "Show that (x - 3) is a factor of x^3 - 6x^2 + 11x - 6", "factor theorem", 1.0, 2.0,
            steps=[
                _step(1, "f(3) = 27 - 54 + 33 - 6 = 0", "partial", 1.0, "Correct calculation but no concluding statement given"),
            ],
            scheme=[_scheme(1, "f(3) = 0, so (x - 3) is a factor", 2.0, "Evaluate f(3) and state the conclusion")],
            feedback=[
                _fb(1, "Correctly substituted x = 3 and calculated f(3) = 0.",
                    missing="A concluding sentence is required: since f(3) = 0, the factor theorem confirms (x - 3) IS a factor.",
                    deduction="1 mark deducted because the calculation alone does not answer the question — the conclusion must be stated explicitly.",
                    improve="Always finish a factor theorem question with an explicit conclusion linking your calculation back to the question."),
            ],
        ),
        # Arithmetic slip solving for unknown coefficient
        _example(
            "If (x - 2) is a factor of x^3 + kx - 4, find k", "factor theorem (unknown coefficient)", 1.0, 2.0,
            steps=[
                _step(1, "f(2) = 8 + 2k - 4 = 0 → 2k = -4", "correct", 1.0),
                _step(2, "k = 2", "incorrect", 0.0, "Sign/arithmetic slip: -4 ÷ 2 = -2, not 2"),
            ],
            scheme=[
                _scheme(1, "2k = -4", 1.0, "Substitute x = 2 and set f(2) = 0"),
                _scheme(2, "k = -2", 1.0, "Solve for k"),
            ],
            feedback=[
                _fb(1, "Correctly substituted x = 2 and set the expression equal to 0."),
                _fb(2, "Correct method of dividing both sides by 2.",
                    missing="-4 ÷ 2 = -2, not 2 — this looks like a sign slip in the division.",
                    deduction="Marks lost for the incorrect final value despite the correct method.",
                    improve="When dividing a negative by a positive, the result stays negative. Verify: 8+2(-2)-4=0 ✓."),
            ],
        ),
        # Incorrect method — wrong substitution value entirely
        _example(
            "Find the remainder when x^3 + 2x^2 - 5 is divided by (x - 2)", "remainder theorem", 0.0, 2.0,
            steps=[
                _step(1, "f(0) = -5", "incorrect", 0.0, "Wrong substitution: for divisor (x-2), the remainder theorem requires evaluating f(2), not f(0)"),
            ],
            scheme=[_scheme(1, "11", 2.0, "Substitute x = 2 into the polynomial")],
            feedback=[
                _fb(1, "You attempted to apply a substitution-based approach.",
                    missing="For divisor (x - 2), the remainder theorem requires evaluating f(2) = 8 + 8 - 5 = 11, not f(0).",
                    deduction="Full marks lost because the wrong value was substituted — the remainder theorem is specific about which value to use.",
                    improve="For a divisor of the form (x - a), always evaluate the polynomial at x = a."),
            ],
        ),
        # Fully correct factor theorem + full factorisation
        _example(
            "Given (x - 1) is a factor of x^3 - 2x^2 - x + 2, factorise fully", "factor theorem", 3.0, 3.0,
            steps=[
                _step(1, "f(1) = 1 - 2 - 1 + 2 = 0, confirming (x - 1) is a factor", "correct", 1.0),
                _step(2, "x^3 - 2x^2 - x + 2 = (x - 1)(x^2 - x - 2)", "correct", 1.0),
                _step(3, "= (x - 1)(x - 2)(x + 1)", "correct", 1.0),
            ],
            scheme=[
                _scheme(1, "f(1) = 0", 1.0, "Confirm (x-1) is a factor using the factor theorem"),
                _scheme(2, "(x - 1)(x^2 - x - 2)", 1.0, "Divide to find the quadratic factor"),
                _scheme(3, "(x - 1)(x - 2)(x + 1)", 1.0, "Fully factorise the remaining quadratic"),
            ],
            feedback=[
                _fb(1, "Correctly confirmed (x - 1) is a factor using the factor theorem."),
                _fb(2, "Correct polynomial division to find the remaining quadratic factor."),
                _fb(3, "Correctly factorised the quadratic to complete the full factorisation.", improve="Full marks — fully factorised."),
            ],
        ),
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Topic: Algebraic fractions
# ─────────────────────────────────────────────────────────────────────────────

def _algebraic_fractions_examples() -> List[Dict]:
    return [
        # Fully correct simplification
        _example(
            "Simplify (x^2 - 9)/(x + 3)", "simplifying algebraic fractions", 2.0, 2.0,
            steps=[
                _step(1, "(x - 3)(x + 3)/(x + 3)", "correct", 1.0),
                _step(2, "= x - 3 (x ≠ -3)", "correct", 1.0),
            ],
            scheme=[
                _scheme(1, "(x - 3)(x + 3)/(x + 3)", 1.0, "Factorise the numerator"),
                _scheme(2, "x - 3, x ≠ -3", 1.0, "Cancel the common factor and state the restriction"),
            ],
            feedback=[
                _fb(1, "Correctly factorised the numerator as a difference of squares."),
                _fb(2, "Correctly cancelled the common factor and stated the restriction.", improve="Full marks."),
            ],
        ),
        # Sign error — dropped negative
        _example(
            "Simplify (4 - x^2)/(x - 2)", "simplifying algebraic fractions", 1.0, 2.0,
            steps=[
                _step(1, "-(x^2 - 4)/(x - 2) = -(x - 2)(x + 2)/(x - 2)", "correct", 1.0),
                _step(2, "= x + 2", "incorrect", 0.0, "Sign dropped — should be -(x + 2)"),
            ],
            scheme=[
                _scheme(1, "-(x - 2)(x + 2)/(x - 2)", 1.0, "Factor out -1 and factorise"),
                _scheme(2, "-(x + 2)", 1.0, "Cancel and keep the negative sign"),
            ],
            feedback=[
                _fb(1, "Correctly rewrote 4 - x^2 as -(x^2 - 4) and factorised it."),
                _fb(2, "You correctly cancelled the common factor (x - 2).",
                    missing="The negative sign from factoring out -1 must be kept in the final answer: -(x + 2), not x + 2.",
                    deduction="1 mark deducted because the negative sign was dropped after cancelling.",
                    improve="When you factor out -1 at the start, carry that negative sign through to your final simplified answer."),
            ],
        ),
        # Incomplete — missing restriction
        _example(
            "Simplify (x^2 + 5x + 6)/(x + 2)", "simplifying algebraic fractions", 1.5, 2.0,
            steps=[
                _step(1, "(x + 2)(x + 3)/(x + 2)", "correct", 1.0),
                _step(2, "= x + 3", "partial", 0.5, "Missing restriction x ≠ -2 (denominator cannot be zero)"),
            ],
            scheme=[
                _scheme(1, "(x + 2)(x + 3)/(x + 2)", 1.0, "Factorise the numerator"),
                _scheme(2, "x + 3, x ≠ -2", 1.0, "Cancel and state the restriction"),
            ],
            feedback=[
                _fb(1, "Correctly factorised the numerator."),
                _fb(2, "Correctly cancelled the common factor to get x + 3.",
                    missing="The restriction x ≠ -2 must be stated, since the original denominator (x + 2) cannot equal zero.",
                    deduction="Half a mark deducted for omitting the domain restriction.",
                    improve="Whenever you cancel a factor from a denominator, state the value(s) of x that must be excluded."),
            ],
        ),
        # Incorrect method — adding fractions by adding across
        _example(
            "Simplify 1/x + 1/(x + 1)", "adding algebraic fractions", 0.0, 2.0,
            steps=[
                _step(1, "= 2/(2x + 1)", "incorrect", 0.0, "Cannot add fractions by adding numerators and denominators separately"),
            ],
            scheme=[_scheme(1, "(2x + 1)/(x(x + 1))", 2.0, "Find a common denominator, then add numerators")],
            feedback=[
                _fb(1, "You attempted to combine the two fractions into one.",
                    missing="Fractions cannot be added by adding numerators and denominators directly. You need a common denominator x(x+1): [1(x+1) + 1(x)] / [x(x+1)] = (2x+1)/(x(x+1)).",
                    deduction="Full marks lost because the fraction addition rule was applied incorrectly.",
                    improve="To add a/b + c/d, use a common denominator: (ad + bc)/(bd). Never add numerators and denominators separately."),
            ],
        ),
        # Correct method, cancellation arithmetic slip
        _example(
            "Simplify (x/2) × (4/x^2)", "multiplying algebraic fractions", 1.0, 2.0,
            steps=[
                _step(1, "= 4x / (2x^2)", "correct", 1.0),
                _step(2, "= 2/x^2", "incorrect", 0.0, "Cancellation slip: only one x cancels, giving 2/x, not 2/x^2"),
            ],
            scheme=[
                _scheme(1, "4x / (2x^2)", 1.0, "Multiply numerators and denominators"),
                _scheme(2, "2/x", 1.0, "Cancel common factors correctly"),
            ],
            feedback=[
                _fb(1, "Correctly multiplied the numerators and denominators."),
                _fb(2, "You attempted to cancel common factors.",
                    missing="4x/(2x^2) = 2/x — one x cancels from numerator and denominator, leaving x^1 in the denominator, not x^2.",
                    deduction="Marks lost for a cancellation slip.",
                    improve="Cancel one matching factor at a time and recount carefully: x/x^2 leaves 1/x, not 1/x^2."),
            ],
        ),
        # Fully correct subtraction with common denominator
        _example(
            "Simplify 3/(x - 1) - 2/(x + 1)", "subtracting algebraic fractions", 2.0, 2.0,
            steps=[
                _step(1, "[3(x + 1) - 2(x - 1)] / [(x - 1)(x + 1)]", "correct", 1.0),
                _step(2, "= (x + 5) / (x^2 - 1)", "correct", 1.0),
            ],
            scheme=[
                _scheme(1, "[3(x+1) - 2(x-1)] / [(x-1)(x+1)]", 1.0, "Use the common denominator (x-1)(x+1)"),
                _scheme(2, "(x + 5) / (x^2 - 1)", 1.0, "Expand and simplify the numerator"),
            ],
            feedback=[
                _fb(1, "Correctly used the common denominator (x-1)(x+1)."),
                _fb(2, "Correctly expanded and collected terms: 3x+3-2x+2 = x+5.", improve="Full marks."),
            ],
        ),
        # Incorrect factorisation of the denominator
        _example(
            "Simplify (x + 4)/(x^2 - 16)", "simplifying algebraic fractions", 0.5, 2.0,
            steps=[
                _step(1, "(x + 4)/[(x - 4)(x - 4)]", "incorrect", 0.5, "Incorrect factorisation: x^2-16 = (x-4)(x+4), not (x-4)(x-4)"),
            ],
            scheme=[_scheme(1, "1/(x - 4)", 2.0, "Factorise the denominator as a difference of squares, then cancel")],
            feedback=[
                _fb(1, "You correctly recognised the denominator needs factorising.",
                    missing="x^2 - 16 is a difference of squares: (x - 4)(x + 4), not (x - 4)(x - 4).",
                    deduction="1.5 marks deducted for the incorrect factorisation of the denominator.",
                    improve="For a^2 - b^2, the factorisation is always (a-b)(a+b) — one plus, one minus, never both the same sign."),
            ],
        ),
        # Fully correct simple cancellation
        _example(
            "Simplify (2x + 4)/(x + 2)", "simplifying algebraic fractions", 1.0, 1.0,
            steps=[
                _step(1, "2(x + 2)/(x + 2) = 2", "correct", 1.0),
            ],
            scheme=[_scheme(1, "2", 1.0, "Factor the numerator and cancel")],
            feedback=[_fb(1, "Correctly factored out 2 from the numerator and cancelled the common bracket.", improve="Full marks.")],
        ),
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Topic: Inequalities
# ─────────────────────────────────────────────────────────────────────────────

def _inequalities_examples() -> List[Dict]:
    return [
        # Fully correct linear inequality
        _example(
            "Solve 2x + 3 > 11", "linear inequality", 1.0, 1.0,
            steps=[
                _step(1, "2x > 8 → x > 4", "correct", 1.0),
            ],
            scheme=[_scheme(1, "x > 4", 1.0, "Subtract 3, then divide by 2")],
            feedback=[_fb(1, "Correctly isolated x — no sign flip needed since dividing by a positive number.", improve="Full marks.")],
        ),
        # Sign-flip error dividing by a negative
        _example(
            "Solve -3x + 6 < 0", "linear inequality", 1.0, 2.0,
            steps=[
                _step(1, "-3x < -6", "correct", 1.0),
                _step(2, "x < 2", "incorrect", 0.0, "Sign error: dividing by a negative number flips the inequality — should be x > 2"),
            ],
            scheme=[
                _scheme(1, "-3x < -6", 1.0, "Subtract 6 from both sides"),
                _scheme(2, "x > 2", 1.0, "Divide by -3 and flip the inequality"),
            ],
            feedback=[
                _fb(1, "Correctly subtracted 6 from both sides."),
                _fb(2, "You correctly divided both sides by -3.",
                    missing="Dividing (or multiplying) an inequality by a negative number flips the inequality sign — the answer should be x > 2, not x < 2.",
                    deduction="Full marks lost for this step because the inequality sign was not flipped.",
                    improve="Always flip the inequality sign whenever you multiply or divide both sides by a negative number."),
            ],
        ),
        # Fully correct quadratic inequality
        _example(
            "Solve x^2 - 5x + 6 > 0", "quadratic inequality", 2.0, 2.0,
            steps=[
                _step(1, "(x - 2)(x - 3) > 0", "correct", 1.0),
                _step(2, "x < 2 or x > 3", "correct", 1.0),
            ],
            scheme=[
                _scheme(1, "(x - 2)(x - 3) > 0", 1.0, "Factorise the quadratic"),
                _scheme(2, "x < 2 or x > 3", 1.0, "Identify the regions where the product is positive"),
            ],
            feedback=[
                _fb(1, "Correct factorisation."),
                _fb(2, "Correctly identified that the product of two factors is positive outside the roots.", improve="Full marks."),
            ],
        ),
        # Quadratic inequality — incomplete (missing one bound)
        _example(
            "Solve x^2 - 4 < 0", "quadratic inequality", 1.0, 2.0,
            steps=[
                _step(1, "(x - 2)(x + 2) < 0", "correct", 1.0),
                _step(2, "x < 2", "partial", 0.0, "Missing lower bound — should be -2 < x < 2"),
            ],
            scheme=[
                _scheme(1, "(x - 2)(x + 2) < 0", 1.0, "Factorise the quadratic"),
                _scheme(2, "-2 < x < 2", 1.0, "Identify the region between the roots"),
            ],
            feedback=[
                _fb(1, "Correct factorisation."),
                _fb(2, "You correctly identified that x = 2 is an upper limit.",
                    missing="The product of two factors is negative only BETWEEN the roots, so the full solution is -2 < x < 2, not just x < 2.",
                    deduction="1 mark deducted for the missing lower bound.",
                    improve="For a quadratic inequality that factorises into two real roots, sketch the parabola or a sign diagram to see which region satisfies the inequality — 'less than zero' means between the roots for an upward parabola."),
            ],
        ),
        # Incorrect method — cross-multiplying without considering sign
        _example(
            "Solve 1/x > 2", "rational inequality", 0.5, 2.0,
            steps=[
                _step(1, "1 > 2x → x < 1/2", "incorrect", 0.5,
                      "Multiplying both sides by x without knowing its sign is invalid — must consider x > 0 and x < 0 separately"),
            ],
            scheme=[_scheme(1, "0 < x < 1/2", 2.0, "Consider the sign of x before multiplying; only x>0 gives valid solutions here")],
            feedback=[
                _fb(1, "You attempted to clear the fraction by multiplying both sides by x.",
                    missing="You cannot multiply an inequality by x without knowing its sign — if x is negative, the inequality direction would flip, and if x<0, 1/x is always negative so it can never exceed 2. The full correct solution restricted to x>0 is 0 < x < 1/2.",
                    deduction="1.5 marks deducted because multiplying by an unknown-sign variable without splitting into cases is not valid.",
                    improve="For inequalities with a variable in the denominator, either split into cases based on the sign of the denominator, or move everything to one side and use a sign diagram."),
            ],
        ),
        # Arithmetic slip
        _example(
            "Solve 5x - 2 ≥ 13", "linear inequality", 0.0, 1.0,
            steps=[
                _step(1, "5x ≥ 15 → x ≥ 2", "incorrect", 0.0, "Arithmetic slip: 15 ÷ 5 = 3, not 2"),
            ],
            scheme=[_scheme(1, "x ≥ 3", 1.0, "Add 2, then divide by 5")],
            feedback=[
                _fb(1, "You correctly added 2 to both sides to get 5x ≥ 15.",
                    missing="15 ÷ 5 = 3, not 2 — this appears to be a division slip.",
                    deduction="Full marks lost for the incorrect final division despite the correct method.",
                    improve="Double check basic divisions before finalising an answer."),
            ],
        ),
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Topic: Sequences and series (arithmetic and geometric)
# ─────────────────────────────────────────────────────────────────────────────

def _sequences_examples() -> List[Dict]:
    return [
        # Fully correct AP — simultaneous equations from nth term formula
        _example(
            "The 3rd term of an AP is 11 and the 7th term is 23. Find the first term and common difference.",
            "arithmetic sequence", 4.0, 4.0,
            steps=[
                _step(1, "a + 2d = 11", "correct", 1.0),
                _step(2, "a + 6d = 23", "correct", 1.0),
                _step(3, "4d = 12 → d = 3", "correct", 1.0),
                _step(4, "a = 11 - 2(3) = 5", "correct", 1.0),
            ],
            scheme=[
                _scheme(1, "a + 2d = 11", 1.0, "Use the nth term formula for the 3rd term"),
                _scheme(2, "a + 6d = 23", 1.0, "Use the nth term formula for the 7th term"),
                _scheme(3, "d = 3", 1.0, "Subtract to eliminate a"),
                _scheme(4, "a = 5", 1.0, "Substitute back to find a"),
            ],
            feedback=[
                _fb(1, "Correctly applied the nth term formula a + (n-1)d for n=3."),
                _fb(2, "Correctly applied the nth term formula for n=7."),
                _fb(3, "Correctly subtracted the equations to eliminate a."),
                _fb(4, "Correctly substituted back to find a.", improve="Full marks. Verify: T7 = 5+6(3) = 23 ✓."),
            ],
        ),
        # Sign error using common difference
        _example(
            "An AP has a = 10, d = -3. Find the 8th term.", "arithmetic sequence", 0.0, 1.0,
            steps=[
                _step(1, "T8 = 10 + 7(3) = 31", "incorrect", 0.0, "Sign error: d = -3, so 7d = -21, not +21"),
            ],
            scheme=[_scheme(1, "T8 = -11", 1.0, "Substitute a=10, n=8, d=-3 into a + (n-1)d")],
            feedback=[
                _fb(1, "You correctly applied the nth term formula structure T_n = a + (n-1)d.",
                    missing="The common difference is d = -3 (negative), so 7d = 7×(-3) = -21, not +21.",
                    deduction="Full marks lost for using the wrong sign for d.",
                    improve="Always substitute the common difference with its correct sign, especially when it's negative. Correct: T8 = 10 + 7(-3) = 10 - 21 = -11."),
            ],
        ),
        # Incomplete AP sum (forgot the n/2 factor)
        _example(
            "Find the sum of the first 10 terms of the AP 2, 5, 8, ...", "arithmetic series", 0.5, 2.0,
            steps=[
                _step(1, "2a + (n-1)d = 2(2) + 9(3) = 31", "partial", 0.5,
                      "Correctly evaluated the bracket but forgot to multiply by n/2"),
            ],
            scheme=[_scheme(1, "S10 = 155", 2.0, "Sn = (n/2)[2a + (n-1)d]")],
            feedback=[
                _fb(1, "You correctly evaluated the bracket [2a + (n-1)d] = 31.",
                    missing="The sum formula is Sn = (n/2)[2a + (n-1)d] — the n/2 factor (here 10/2=5) was never applied: S10 = 5 × 31 = 155.",
                    deduction="1.5 marks deducted because the final multiplication by n/2 was omitted, leaving the answer incomplete.",
                    improve="Always write out the full sum formula Sn = (n/2)[2a+(n-1)d] and make sure you apply every part of it, including the n/2 factor."),
            ],
        ),
        # Incorrect method — used AP formula for a GP
        _example(
            "Find the 6th term of the GP 3, 6, 12, ...", "geometric sequence", 0.0, 2.0,
            steps=[
                _step(1, "T6 = 3 + 5(3) = 18", "incorrect", 0.0,
                      "Incorrect method: this is a geometric sequence (common ratio), not arithmetic — the AP formula does not apply"),
            ],
            scheme=[_scheme(1, "T6 = 96", 2.0, "T_n = a r^(n-1), with a=3, r=2")],
            feedback=[
                _fb(1, "You correctly identified that a pattern needs to be found between consecutive terms.",
                    missing="This sequence has a common RATIO (each term is 2× the previous), not a common difference — it's geometric, not arithmetic. The correct formula is T_n = a r^(n-1): T6 = 3 × 2^5 = 96.",
                    deduction="Full marks lost because the wrong type of sequence formula (arithmetic) was applied to a geometric sequence.",
                    improve="Before choosing a formula, check whether consecutive terms share a common DIFFERENCE (arithmetic) or a common RATIO (geometric)."),
            ],
        ),
        # Correct method, arithmetic slip in power calculation
        _example(
            "Find the sum of the first 5 terms of the GP 2, 6, 18, ...", "geometric series", 1.0, 2.0,
            steps=[
                _step(1, "S5 = 2(3^5 - 1)/(3 - 1)", "correct", 1.0),
                _step(2, "3^5 = 343, so S5 = 2(342)/2 = 342", "incorrect", 0.0,
                      "Arithmetic slip: 3^5 = 243, not 343"),
            ],
            scheme=[
                _scheme(1, "S5 = 2(3^5 - 1)/(3 - 1)", 1.0, "Apply the geometric series sum formula with a=2, r=3"),
                _scheme(2, "S5 = 242", 1.0, "Correctly evaluate 3^5 = 243"),
            ],
            feedback=[
                _fb(1, "Correctly applied the geometric series sum formula Sn = a(r^n-1)/(r-1)."),
                _fb(2, "Correct method of substituting into the formula.",
                    missing="3^5 = 243, not 343 — this appears to be an arithmetic slip in computing the power.",
                    deduction="Full marks lost because the wrong power value led to an incorrect final sum (should be 242, not 342).",
                    improve="Compute powers step by step to avoid slips: 3^2=9, 3^3=27, 3^4=81, 3^5=243."),
            ],
        ),
        # Fully correct GP nth term
        _example(
            "Find the 7th term of the GP 5, 10, 20, ...", "geometric sequence", 1.0, 1.0,
            steps=[
                _step(1, "T7 = 5 × 2^6 = 320", "correct", 1.0),
            ],
            scheme=[_scheme(1, "320", 1.0, "T_n = a r^(n-1), with a=5, r=2")],
            feedback=[_fb(1, "Correctly identified r=2 and applied the nth term formula.", improve="Full marks.")],
        ),
        # Incomplete — sum to infinity left unsimplified
        _example(
            "Find the sum to infinity of the GP 8, 4, 2, ...", "geometric series (sum to infinity)", 0.5, 1.5,
            steps=[
                _step(1, "S∞ = 8/(1 - 0.5) = 8/0.5", "partial", 0.5, "Final division not carried out — should be simplified to 16"),
            ],
            scheme=[_scheme(1, "S∞ = 16", 1.5, "S∞ = a/(1-r), with a=8, r=1/2")],
            feedback=[
                _fb(1, "Correctly identified r = 1/2 and set up the sum-to-infinity formula.",
                    missing="8 ÷ 0.5 must be evaluated to a final number: 8/0.5 = 16.",
                    deduction="1 mark deducted because the division was left incomplete.",
                    improve="Always finish the arithmetic to a final simplified number — 8/0.5 = 16."),
            ],
        ),
        # Fully correct AP sum
        _example(
            "Find the sum of the first 20 terms of the AP 4, 7, 10, ...", "arithmetic series", 2.0, 2.0,
            steps=[
                _step(1, "S20 = (20/2)[2(4) + 19(3)]", "correct", 1.0),
                _step(2, "= 10[8 + 57] = 10(65) = 650", "correct", 1.0),
            ],
            scheme=[
                _scheme(1, "S20 = (20/2)[2(4) + 19(3)]", 1.0, "Apply the arithmetic series sum formula with a=4, d=3, n=20"),
                _scheme(2, "S20 = 650", 1.0, "Evaluate the expression"),
            ],
            feedback=[
                _fb(1, "Correctly applied the sum formula Sn = (n/2)[2a + (n-1)d]."),
                _fb(2, "Correctly evaluated the arithmetic to reach the final sum.", improve="Full marks."),
            ],
        ),
    ]


def create_sample_annotations(output_file: str):
    """
    Create starter raw annotations covering the A/L algebra syllabus —
    linear equations, simultaneous equations, quadratics (factorisation,
    quadratic formula, completing the square), indices & surds, polynomials,
    remainder/factor theorem, algebraic fractions, inequalities, and
    sequences/series — each mixing the 5 error families (sign errors,
    missing steps/roots, incomplete simplification, incorrect method/
    factorisation, correct-method arithmetic slips) with fully-correct
    examples for balanced training.
    """
    samples = (
        _linear_examples()
        + _simultaneous_examples()
        + _quadratic_examples()
        + _indices_surds_examples()
        + _polynomial_examples()
        + _remainder_factor_theorem_examples()
        + _algebraic_fractions_examples()
        + _inequalities_examples()
        + _sequences_examples()
    )

    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(samples, f, indent=2, ensure_ascii=False)

    print(f"Sample annotations created ({len(samples)} examples) → {output_file}")


if __name__ == "__main__":
    RAW_PATH = "app/training/data/raw_annotations.json"
    create_sample_annotations(RAW_PATH)

    with open(RAW_PATH, "r", encoding="utf-8") as f:
        raw = json.load(f)
    formatted = [format_example(item) for item in raw]
    train_set, eval_set = split_dataset(formatted)
    _write_json(train_set, "app/training/data/feedback_dataset_train.json")
    _write_json(eval_set, "app/training/data/feedback_dataset_eval.json")
