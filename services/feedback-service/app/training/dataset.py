"""
Dataset preparation for LoRA fine-tuning of the feedback generation module.

Raw annotation schema (raw_annotations.json):
[
  {
    "id": "quadratic_factorisation-0001",
    "topic": "quadratic_factorisation",
    "difficulty": "medium",           // "easy" | "medium" | "difficult"
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

# Fixed vocabularies for raw-annotation metadata — single source of truth used
# by _example() (construction-time validation) and validate_dataset.py
# (post-hoc validation). Extend this list when adding a new topic's builder
# function; never rename existing values once examples reference them.
TOPICS = (
    "simplifying_expressions",
    "collecting_like_terms",
    "expanding_brackets",
    "factorising_common_factor",
    "factorising_quadratic",
    "difference_of_squares",
    "sum_difference_of_cubes",
    "linear_equations",
    "equations_with_brackets",
    "equations_with_fractions",
    "linear_inequalities",
    "quadratic_inequalities",
    "simultaneous_substitution",
    "simultaneous_elimination",
    "simultaneous_linear_quadratic",
    "quadratic_factorisation",
    "quadratic_formula",
    "completing_the_square",
    "algebraic_fractions",
    "polynomial_addition_subtraction",
    "polynomial_multiplication",
    "polynomial_division",
    "remainder_factor_theorem",
    "index_laws",
    "negative_fractional_indices",
    "surds",
    "rationalising_denominators",
    "functions",
    "function_composition",
    "inverse_functions",
    "straight_line_equations",
    "gradient_intercept",
    "sequences_nth_term",
    "arithmetic_sequences",
    "geometric_sequences",
    "algebraic_word_problems",
    "ratio_proportion",
    "rearranging_formulas",
    "absolute_value_equations",
    "exponent_equations",
)

DIFFICULTIES = ("easy", "medium", "difficult")


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


def split_raw_dataset(
    raw_examples: List[Dict], eval_ratio: float = 0.15, seed: int = 42
) -> Tuple[List[Dict], List[Dict]]:
    """Topic-grouped, stratified split of RAW (pre-format_example) annotations.

    Splitting is done per topic group rather than globally so that every
    topic is represented in both train and eval in roughly the target ratio,
    and so near-identical problem families (which tend to cluster within the
    same topic in this generator) don't straddle the split. Formatting to
    {"text": ...} happens after splitting, on each half separately.
    """
    by_topic: Dict[str, List[Dict]] = {}
    for ex in raw_examples:
        by_topic.setdefault(ex["topic"], []).append(ex)

    train: List[Dict] = []
    eval_: List[Dict] = []
    for topic in sorted(by_topic):
        group = by_topic[topic][:]
        rng = random.Random(f"{seed}:{topic}")
        rng.shuffle(group)
        n_eval = round(len(group) * eval_ratio)
        # Never send a topic's only example(s) entirely to eval, and never
        # split a group of 1 (nothing to stratify).
        if len(group) <= 1:
            n_eval = 0
        n_eval = min(n_eval, len(group) - 1) if len(group) > 1 else 0
        eval_.extend(group[:n_eval])
        train.extend(group[n_eval:])
    return train, eval_


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
    *,
    topic: str,
    difficulty: str,
) -> Dict:
    if topic not in TOPICS:
        raise ValueError(f"Unknown topic {topic!r} for question: {question!r}")
    if difficulty not in DIFFICULTIES:
        raise ValueError(f"Unknown difficulty {difficulty!r} for question: {question!r}")
    return {
        # "id" is left as a placeholder here and assigned centrally in
        # create_sample_annotations() once the full example count is known,
        # so uniqueness is guaranteed mechanically rather than by hand.
        "id": None,
        "topic": topic,
        "difficulty": difficulty,
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
            topic="linear_equations", difficulty="easy",
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
            topic="linear_equations", difficulty="medium",
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
            topic="linear_equations", difficulty="easy",
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
            topic="equations_with_brackets", difficulty="medium",
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
            topic="linear_equations", difficulty="easy",
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
            topic="equations_with_fractions", difficulty="medium",
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
            topic="simultaneous_elimination", difficulty="easy",
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
            topic="simultaneous_elimination", difficulty="medium",
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
            topic="simultaneous_linear_quadratic", difficulty="medium",
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
            topic="simultaneous_linear_quadratic", difficulty="difficult",
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
            topic="simultaneous_substitution", difficulty="easy",
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
            topic="simultaneous_elimination", difficulty="easy",
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
            topic="simultaneous_linear_quadratic", difficulty="difficult",
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
            topic="simultaneous_linear_quadratic", difficulty="difficult",
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
            topic="quadratic_factorisation", difficulty="medium",
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
            topic="difference_of_squares", difficulty="easy",
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
            topic="difference_of_squares", difficulty="easy",
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
            topic="quadratic_factorisation", difficulty="medium",
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
            topic="quadratic_factorisation", difficulty="medium",
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
            topic="quadratic_formula", difficulty="medium",
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
            topic="quadratic_factorisation", difficulty="easy",
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
            topic="quadratic_formula", difficulty="medium",
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
            topic="completing_the_square", difficulty="medium",
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
            topic="completing_the_square", difficulty="difficult",
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
            topic="index_laws", difficulty="easy",
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
            topic="index_laws", difficulty="medium",
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
            topic="index_laws", difficulty="easy",
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
            topic="negative_fractional_indices", difficulty="difficult",
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
            topic="index_laws", difficulty="easy",
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
            topic="surds", difficulty="easy",
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
            topic="rationalising_denominators", difficulty="medium",
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
            topic="surds", difficulty="easy",
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
            topic="expanding_brackets", difficulty="easy",
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
            topic="expanding_brackets", difficulty="easy",
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
            topic="factorising_common_factor", difficulty="easy",
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
            topic="factorising_common_factor", difficulty="medium",
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
            topic="sum_difference_of_cubes", difficulty="difficult",
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
            topic="sum_difference_of_cubes", difficulty="difficult",
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
            topic="factorising_common_factor", difficulty="medium",
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
            topic="factorising_common_factor", difficulty="medium",
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
            topic="remainder_factor_theorem", difficulty="medium",
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
            topic="remainder_factor_theorem", difficulty="medium",
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
            topic="remainder_factor_theorem", difficulty="medium",
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
            topic="remainder_factor_theorem", difficulty="difficult",
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
            topic="remainder_factor_theorem", difficulty="medium",
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
            topic="remainder_factor_theorem", difficulty="difficult",
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
            topic="algebraic_fractions", difficulty="easy",
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
            topic="algebraic_fractions", difficulty="medium",
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
            topic="algebraic_fractions", difficulty="medium",
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
            topic="algebraic_fractions", difficulty="medium",
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
            topic="algebraic_fractions", difficulty="easy",
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
            topic="algebraic_fractions", difficulty="medium",
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
            topic="algebraic_fractions", difficulty="medium",
        ),
        # Fully correct simple cancellation
        _example(
            "Simplify (2x + 4)/(x + 2)", "simplifying algebraic fractions", 1.0, 1.0,
            steps=[
                _step(1, "2(x + 2)/(x + 2) = 2", "correct", 1.0),
            ],
            scheme=[_scheme(1, "2", 1.0, "Factor the numerator and cancel")],
            feedback=[_fb(1, "Correctly factored out 2 from the numerator and cancelled the common bracket.", improve="Full marks.")],
            topic="algebraic_fractions", difficulty="easy",
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
            topic="linear_inequalities", difficulty="easy",
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
            topic="linear_inequalities", difficulty="medium",
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
            topic="quadratic_inequalities", difficulty="medium",
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
            topic="quadratic_inequalities", difficulty="medium",
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
            topic="quadratic_inequalities", difficulty="difficult",
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
            topic="linear_inequalities", difficulty="easy",
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
            topic="arithmetic_sequences", difficulty="medium",
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
            topic="arithmetic_sequences", difficulty="easy",
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
            topic="arithmetic_sequences", difficulty="medium",
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
            topic="geometric_sequences", difficulty="medium",
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
            topic="geometric_sequences", difficulty="medium",
        ),
        # Fully correct GP nth term
        _example(
            "Find the 7th term of the GP 5, 10, 20, ...", "geometric sequence", 1.0, 1.0,
            steps=[
                _step(1, "T7 = 5 × 2^6 = 320", "correct", 1.0),
            ],
            scheme=[_scheme(1, "320", 1.0, "T_n = a r^(n-1), with a=5, r=2")],
            feedback=[_fb(1, "Correctly identified r=2 and applied the nth term formula.", improve="Full marks.")],
            topic="geometric_sequences", difficulty="easy",
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
            topic="geometric_sequences", difficulty="medium",
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
            topic="arithmetic_sequences", difficulty="medium",
        ),
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Topic: Simplifying expressions
# ─────────────────────────────────────────────────────────────────────────────

def _simplifying_expressions_examples() -> List[Dict]:
    return [
        _example(
            "Simplify 5x + 3y - 2x + 7y", "simplifying expressions", 1.0, 1.0,
            steps=[
                _step(1, "5x - 2x = 3x", "correct", 0.5),
                _step(2, "3y + 7y = 10y", "correct", 0.5),
            ],
            scheme=[
                _scheme(1, "3x", 0.5, "Combine the x terms"),
                _scheme(2, "10y", 0.5, "Combine the y terms"),
            ],
            feedback=[
                _fb(1, "Correctly combined the x terms: 5x - 2x = 3x."),
                _fb(2, "Correctly combined the y terms: 3y + 7y = 10y.",
                    improve="Full marks. Final answer: 3x + 10y."),
            ],
            topic="simplifying_expressions", difficulty="easy",
        ),
        _example(
            "Simplify 4x + 3x^2 - x", "simplifying expressions", 1.0, 2.0,
            steps=[
                _step(1, "4x - x = 3x", "correct", 1.0),
                _step(2, "3x^2 + 3x = 6x^2", "incorrect", 0.0,
                      "Combining unlike terms: 3x^2 and 3x have different powers of x"),
            ],
            scheme=[
                _scheme(1, "3x", 1.0, "Combine the x terms (4x - x)"),
                _scheme(2, "3x^2 + 3x", 1.0, "Leave unlike terms separate — final answer 3x^2 + 3x"),
            ],
            feedback=[
                _fb(1, "You correctly combined the x terms: 4x - x = 3x."),
                _fb(2, "You attempted to combine your remaining terms.",
                    missing="3x^2 and 3x have different powers of x (x^2 vs x^1), so they are NOT like terms and cannot be combined. The final simplified answer is 3x^2 + 3x.",
                    deduction="Full marks lost because unlike terms were combined as if they were like terms.",
                    improve="Only combine terms that have exactly the same variable and exponent. Different powers of x are different terms."),
            ],
            topic="simplifying_expressions", difficulty="medium",
        ),
        _example(
            "Simplify 6x - 2(x + 3)", "simplifying expressions", 1.0, 2.0,
            steps=[
                _step(1, "6x - 2x - 6", "correct", 1.0),
                _step(2, "3x - 6", "incorrect", 0.0, "Arithmetic slip: 6x - 2x = 4x, not 3x"),
            ],
            scheme=[
                _scheme(1, "6x - 2x - 6", 1.0, "Distribute -2 across the bracket"),
                _scheme(2, "4x - 6", 1.0, "Combine the x terms"),
            ],
            feedback=[
                _fb(1, "You correctly distributed the -2 across the bracket: -2(x+3) = -2x - 6."),
                _fb(2, "Correct method of combining the x terms.",
                    missing="6x - 2x = 4x, not 3x — this looks like an arithmetic slip.",
                    deduction="1 mark lost for the incorrect combination of the x terms despite the correct distribution.",
                    improve="Double-check basic subtraction: 6x - 2x = 4x. Final answer: 4x - 6."),
            ],
            topic="simplifying_expressions", difficulty="medium",
        ),
        _example(
            "Simplify 3(2a - 5) + 4a", "simplifying expressions", 2.0, 2.0,
            steps=[
                _step(1, "6a - 15 + 4a", "correct", 1.0),
                _step(2, "10a - 15", "correct", 1.0),
            ],
            scheme=[
                _scheme(1, "6a - 15 + 4a", 1.0, "Distribute the 3 across the bracket"),
                _scheme(2, "10a - 15", 1.0, "Combine like terms"),
            ],
            feedback=[
                _fb(1, "Correctly distributed the 3 across the bracket."),
                _fb(2, "Correctly combined the a terms: 6a + 4a = 10a.", improve="Full marks."),
            ],
            topic="simplifying_expressions", difficulty="easy",
        ),
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Topic: Collecting like terms
# ─────────────────────────────────────────────────────────────────────────────

def _collecting_like_terms_examples() -> List[Dict]:
    return [
        _example(
            "Simplify 7a + 4b - 3a + 2b", "collecting like terms", 1.0, 1.0,
            steps=[
                _step(1, "7a - 3a = 4a", "correct", 0.5),
                _step(2, "4b + 2b = 6b", "correct", 0.5),
            ],
            scheme=[
                _scheme(1, "4a", 0.5, "Combine the a terms"),
                _scheme(2, "6b", 0.5, "Combine the b terms"),
            ],
            feedback=[
                _fb(1, "Correctly combined the a terms."),
                _fb(2, "Correctly combined the b terms.", improve="Full marks. Final answer: 4a + 6b."),
            ],
            topic="collecting_like_terms", difficulty="easy",
        ),
        _example(
            "Simplify 5m + 3n + 2m - n", "collecting like terms", 0.0, 1.0,
            steps=[
                _step(1, "5m + 3n + 2m - n = 9mn", "incorrect", 0.0,
                      "Combined unlike terms m and n into a single term"),
            ],
            scheme=[_scheme(1, "7m + 2n", 1.0, "Combine m terms and n terms separately")],
            feedback=[
                _fb(1, "You attempted to simplify by combining terms.",
                    missing="m terms and n terms are different variables and must be combined separately: 5m+2m=7m and 3n-n=2n, giving 7m+2n, not a single combined term 9mn.",
                    deduction="Full marks lost because unlike terms (m and n) were incorrectly combined into one term.",
                    improve="Only combine terms with the exact same variable(s). Group m terms together and n terms together separately."),
            ],
            topic="collecting_like_terms", difficulty="easy",
        ),
        _example(
            "Simplify 8p - 5q - 3p + 5q", "collecting like terms", 1.0, 2.0,
            steps=[
                _step(1, "8p - 3p = 5p", "correct", 1.0),
                _step(2, "-5q + 5q = -10q", "incorrect", 0.0,
                      "A negative and its positive counterpart cancel to 0, not -10q"),
            ],
            scheme=[
                _scheme(1, "5p", 1.0, "Combine the p terms"),
                _scheme(2, "0 (the q terms cancel)", 1.0, "Combine the q terms"),
            ],
            feedback=[
                _fb(1, "You correctly combined the p terms: 8p - 3p = 5p."),
                _fb(2, "You correctly identified this step involves combining the q terms.",
                    missing="-5q + 5q = 0 (a negative and its positive counterpart cancel to zero), not -10q.",
                    deduction="Full marks lost because the q terms were combined incorrectly — they should cancel to zero.",
                    improve="When adding a negative and positive of the same term with equal coefficients, they cancel to 0. Final answer: 5p (the q term vanishes)."),
            ],
            topic="collecting_like_terms", difficulty="medium",
        ),
        _example(
            "Simplify 4x^2 + 3x - 2x^2 + 5x - 7", "collecting like terms", 1.5, 2.0,
            steps=[
                _step(1, "4x^2 - 2x^2 = 2x^2", "correct", 0.5),
                _step(2, "3x + 5x = 8x", "correct", 0.5),
                _step(3, "2x^2 + 8x", "partial", 0.5, "Missing the constant term — should be 2x^2 + 8x - 7"),
            ],
            scheme=[
                _scheme(1, "2x^2", 0.5, "Combine the x^2 terms"),
                _scheme(2, "8x", 0.5, "Combine the x terms"),
                _scheme(3, "2x^2 + 8x - 7", 1.0, "Include the constant term in the final answer"),
            ],
            feedback=[
                _fb(1, "Correctly combined the x^2 terms."),
                _fb(2, "Correctly combined the x terms."),
                _fb(3, "You correctly combined the x^2 and x terms.",
                    missing="The constant term -7 has no like term to combine with, so it must still appear in the final answer: 2x^2 + 8x - 7.",
                    deduction="0.5 mark deducted because the constant term was dropped from the final simplified expression.",
                    improve="Terms with no matching like term still carry through to the final answer unchanged — don't drop them."),
            ],
            topic="collecting_like_terms", difficulty="medium",
        ),
        _example(
            "Simplify 9a - 4a + 6b - 2b", "collecting like terms", 1.0, 1.0,
            steps=[
                _step(1, "9a - 4a = 5a", "correct", 0.5),
                _step(2, "6b - 2b = 4b", "correct", 0.5),
            ],
            scheme=[
                _scheme(1, "5a", 0.5, "Combine the a terms"),
                _scheme(2, "4b", 0.5, "Combine the b terms"),
            ],
            feedback=[
                _fb(1, "Correctly combined the a terms."),
                _fb(2, "Correctly combined the b terms.", improve="Full marks. Final answer: 5a + 4b."),
            ],
            topic="collecting_like_terms", difficulty="easy",
        ),
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Topic: Factorising quadratic expressions (not equations — no "=0")
# ─────────────────────────────────────────────────────────────────────────────

def _factorising_quadratic_examples() -> List[Dict]:
    return [
        _example(
            "Factorise x^2 + 7x + 12", "factorisation", 2.0, 2.0,
            steps=[_step(1, "(x + 3)(x + 4)", "correct", 2.0)],
            scheme=[_scheme(1, "(x + 3)(x + 4)", 2.0, "Find a factor pair of 12 that sums to 7: 3 and 4")],
            feedback=[_fb(1, "Correct — 3 x 4 = 12 and 3 + 4 = 7.", improve="Full marks.")],
            topic="factorising_quadratic", difficulty="easy",
        ),
        _example(
            "Factorise x^2 - 2x - 15", "factorisation", 0.0, 2.0,
            steps=[
                _step(1, "(x + 5)(x - 3)", "incorrect", 0.0,
                      "Needs factors of -15 that sum to -2: -5 and 3, not +5 and -3"),
            ],
            scheme=[_scheme(1, "(x - 5)(x + 3)", 2.0, "Find a factor pair of -15 that sums to -2: -5 and 3")],
            feedback=[
                _fb(1, "You correctly identified 5 and 3 as a factor pair of 15.",
                    missing="You need (-5) + (3) = -2, but your factors (+5) and (-3) sum to +2, giving the wrong sign. Correct factorisation: (x - 5)(x + 3).",
                    deduction="Full marks lost — (x+5)(x-3) expands to x^2+2x-15, which does not match x^2-2x-15.",
                    improve="Always verify a factorisation by expanding it back out and checking it matches the original expression."),
            ],
            topic="factorising_quadratic", difficulty="medium",
        ),
        _example(
            "Factorise 2x^2 + 7x + 3", "factorisation by grouping", 3.0, 4.0,
            steps=[
                _step(1, "2x^2 + 6x + x + 3", "correct", 1.0),
                _step(2, "2x(x + 3) + 1(x + 3)", "correct", 1.0),
                _step(3, "2x(x + 3) + (x + 3)", "partial", 1.0,
                      "Final step not completed — should be written as (2x + 1)(x + 3)"),
            ],
            scheme=[
                _scheme(1, "2x^2 + 6x + x + 3", 1.0, "Split the middle term using factors of 2x3=6 that sum to 7: 6 and 1"),
                _scheme(2, "2x(x + 3) + 1(x + 3)", 1.0, "Factor by grouping"),
                _scheme(3, "(2x + 1)(x + 3)", 2.0, "Factor out the common bracket (x + 3)"),
            ],
            feedback=[
                _fb(1, "Correctly split the middle term using 6 and 1, factors of 6 that sum to 7."),
                _fb(2, "Correctly factored each pair of terms, revealing the common bracket (x + 3)."),
                _fb(3, "You correctly factored out (x + 3) from both grouped terms.",
                    missing="The final step is to write this as a single product: (2x + 1)(x + 3), combining the leftover 2x and 1 as the other factor.",
                    deduction="1 mark deducted because the factorisation was not written in its final fully-factored form.",
                    improve="After grouping and factoring out the common bracket, always write the remaining terms as the second factor: (2x + 1)(x + 3)."),
            ],
            topic="factorising_quadratic", difficulty="difficult",
        ),
        _example(
            "Factorise x^2 - 6x + 9", "factorisation (perfect square)", 2.0, 2.0,
            steps=[_step(1, "(x - 3)^2", "correct", 2.0)],
            scheme=[_scheme(1, "(x - 3)^2", 2.0, "Recognise as a perfect square: half of -6 is -3, and (-3)^2 = 9")],
            feedback=[_fb(1, "Correct — (x-3)^2 = x^2-6x+9.", improve="Full marks — well spotted as a perfect square.")],
            topic="factorising_quadratic", difficulty="medium",
        ),
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Topic: Polynomial addition and subtraction
# ─────────────────────────────────────────────────────────────────────────────

def _polynomial_addition_subtraction_examples() -> List[Dict]:
    return [
        _example(
            "Add (3x^2 + 2x - 5) and (x^2 - 4x + 7)", "polynomial addition", 1.0, 1.0,
            steps=[_step(1, "4x^2 - 2x + 2", "correct", 1.0)],
            scheme=[_scheme(1, "4x^2 - 2x + 2", 1.0, "Add corresponding like terms")],
            feedback=[_fb(1, "Correctly added corresponding like terms: 3x^2+x^2=4x^2, 2x-4x=-2x, -5+7=2.",
                           improve="Full marks.")],
            topic="polynomial_addition_subtraction", difficulty="easy",
        ),
        _example(
            "Subtract (2x^2 - 3x + 4) from (5x^2 + x - 1)", "polynomial subtraction", 0.0, 2.0,
            steps=[
                _step(1, "5x^2 + x - 1 - 2x^2 - 3x + 4", "incorrect", 0.0,
                      "Only the first term's sign was flipped when distributing the subtraction"),
                _step(2, "3x^2 - 2x + 3", "incorrect", 0.0, "Follows from the sign error in step 1"),
            ],
            scheme=[
                _scheme(1, "5x^2 + x - 1 - 2x^2 + 3x - 4", 1.0, "Distribute the negative sign to every term"),
                _scheme(2, "3x^2 + 4x - 5", 1.0, "Combine like terms"),
            ],
            feedback=[
                _fb(1, "You correctly set up the subtraction and flipped the sign of the first term (2x^2 → -2x^2).",
                    missing="The negative sign must be distributed to ALL three terms in the bracket: -(2x^2-3x+4) = -2x^2+3x-4, not -2x^2-3x+4.",
                    deduction="Full marks lost because two of the three signs were not flipped."),
                _fb(2, "Your combination of like terms is arithmetically consistent with your Step 1.",
                    missing="Because Step 1 had the sign error, this final answer is wrong. The correct answer is 3x^2+4x-5.",
                    deduction="Marks lost because this follows directly from the earlier sign error.",
                    improve="When subtracting a polynomial, first rewrite it with every sign flipped, THEN combine like terms."),
            ],
            topic="polynomial_addition_subtraction", difficulty="medium",
        ),
        _example(
            "Add (4x^3 - 2x + 5) and (-x^3 + 3x^2 - 5)", "polynomial addition", 1.0, 1.0,
            steps=[_step(1, "3x^3 + 3x^2 - 2x", "correct", 1.0)],
            scheme=[_scheme(1, "3x^3 + 3x^2 - 2x", 1.0, "Add corresponding like terms")],
            feedback=[_fb(1, "Correctly combined all like terms — the constant terms cancel to 0 (5-5=0) so don't appear in the final answer.",
                           improve="Full marks.")],
            topic="polynomial_addition_subtraction", difficulty="easy",
        ),
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Topic: Polynomial multiplication
# ─────────────────────────────────────────────────────────────────────────────

def _polynomial_multiplication_examples() -> List[Dict]:
    return [
        _example(
            "Multiply (x + 3)(x^2 - 2x + 5)", "polynomial multiplication", 3.0, 3.0,
            steps=[
                _step(1, "x^3 - 2x^2 + 5x + 3x^2 - 6x + 15", "correct", 1.5),
                _step(2, "x^3 + x^2 - x + 15", "correct", 1.5),
            ],
            scheme=[
                _scheme(1, "x^3-2x^2+5x+3x^2-6x+15", 1.5, "Distribute each term of the first bracket across the second"),
                _scheme(2, "x^3 + x^2 - x + 15", 1.5, "Combine like terms"),
            ],
            feedback=[
                _fb(1, "Correctly distributed x and 3 across the second bracket."),
                _fb(2, "Correctly combined like terms: -2x^2+3x^2=x^2, 5x-6x=-x.", improve="Full marks."),
            ],
            topic="polynomial_multiplication", difficulty="medium",
        ),
        _example(
            "Multiply (2x - 1)(x + 4)", "polynomial multiplication", 1.0, 2.0,
            steps=[
                _step(1, "2x^2 + 8x - x - 4", "correct", 1.0),
                _step(2, "2x^2 + 6x - 4", "incorrect", 0.0, "Arithmetic slip: 8x - x = 7x, not 6x"),
            ],
            scheme=[
                _scheme(1, "2x^2 + 8x - x - 4", 1.0, "Expand using FOIL"),
                _scheme(2, "2x^2 + 7x - 4", 1.0, "Combine like terms"),
            ],
            feedback=[
                _fb(1, "Correct expansion using FOIL."),
                _fb(2, "Correct method of combining like terms.",
                    missing="8x - x = 7x, not 6x — this looks like an arithmetic slip.",
                    deduction="Marks lost for the incorrect combination despite the correct expansion.",
                    improve="Double-check basic subtraction. Final answer: 2x^2 + 7x - 4."),
            ],
            topic="polynomial_multiplication", difficulty="easy",
        ),
        _example(
            "Multiply (x - 5)^2", "polynomial multiplication (squaring a bracket)", 0.0, 2.0,
            steps=[
                _step(1, "x^2 - 25", "incorrect", 0.0,
                      "(x-5)^2 is not a difference of squares — the middle cross-term -2ab was omitted entirely"),
            ],
            scheme=[_scheme(1, "x^2 - 10x + 25", 2.0, "Expand (x-5)^2 = x^2 - 2(x)(5) + 5^2")],
            feedback=[
                _fb(1, "You correctly squared the first and last terms (x^2 and 25).",
                    missing="(x-5)^2 is NOT the same as difference of squares — it must be expanded as (x-5)(x-5) = x^2-2(x)(5)+25 = x^2-10x+25. The middle term -10x was left out entirely.",
                    deduction="Full marks lost because the middle cross-term was omitted — (a-b)^2 ≠ a^2-b^2.",
                    improve="Always expand (a±b)^2 fully as (a±b)(a±b), or memorise (a±b)^2=a^2±2ab+b^2 — never skip the middle term."),
            ],
            topic="polynomial_multiplication", difficulty="medium",
        ),
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Topic: Polynomial division
# ─────────────────────────────────────────────────────────────────────────────

def _polynomial_division_examples() -> List[Dict]:
    return [
        _example(
            "Divide (6x^3 - 9x^2 + 3x) by 3x", "polynomial division", 2.0, 2.0,
            steps=[
                _step(1, "6x^3/3x - 9x^2/3x + 3x/3x", "correct", 1.0),
                _step(2, "2x^2 - 3x + 1", "correct", 1.0),
            ],
            scheme=[
                _scheme(1, "6x^3/3x - 9x^2/3x + 3x/3x", 1.0, "Divide each term by 3x"),
                _scheme(2, "2x^2 - 3x + 1", 1.0, "Simplify each quotient"),
            ],
            feedback=[
                _fb(1, "Correctly divided each term of the numerator by 3x."),
                _fb(2, "Correctly simplified each term.", improve="Full marks."),
            ],
            topic="polynomial_division", difficulty="easy",
        ),
        _example(
            "Divide (x^2 + 5x + 6) by (x + 2)", "polynomial long division", 1.0, 3.0,
            steps=[
                _step(1, "x(x + 2) = x^2 + 2x", "correct", 1.0),
                _step(2, "(x^2 + 5x + 6) - (x^2 + 2x) = 5x + 6", "incorrect", 0.0,
                      "Subtraction error: 5x - 2x = 3x, not 5x — the 2x term was not subtracted"),
                _step(3, "Quotient continues to x + 5", "incorrect", 0.0, "Follows from the subtraction error in step 2"),
            ],
            scheme=[
                _scheme(1, "x^2 + 2x", 1.0, "Multiply the first quotient term x by the divisor"),
                _scheme(2, "3x + 6", 1.0, "Subtract correctly"),
                _scheme(3, "x + 3, remainder 0", 1.0, "Complete the division"),
            ],
            feedback=[
                _fb(1, "Correctly multiplied x by the divisor (x+2)."),
                _fb(2, "You correctly set up the subtraction of x^2+2x from the original polynomial.",
                    missing="(x^2+5x+6)-(x^2+2x) = (5x-2x)+6 = 3x+6, not 5x+6 — the 2x must be subtracted from the 5x term.",
                    deduction="Full marks lost for this subtraction error."),
                _fb(3, "Your method of continuing the division process is correct.",
                    missing="Because Step 2 was wrong, this quotient is wrong. Using the correct remainder 3x+6, dividing by x+2 gives exactly 3, so the full quotient is x+3 with remainder 0.",
                    deduction="Marks lost because this follows from the earlier subtraction error.",
                    improve="In polynomial long division, subtract the ENTIRE product (all terms) from the corresponding terms of the dividend, not just partially."),
            ],
            topic="polynomial_division", difficulty="difficult",
        ),
        _example(
            "Divide (2x^2 + 7x + 3) by (2x + 1)", "polynomial division (by factorisation)", 2.0, 2.0,
            steps=[
                _step(1, "(2x + 1)(x + 3) = 2x^2 + 7x + 3", "correct", 1.0),
                _step(2, "quotient = x + 3", "correct", 1.0),
            ],
            scheme=[
                _scheme(1, "(2x+1)(x+3)=2x^2+7x+3", 1.0, "Factorise the dividend, recognising (2x+1) as a factor"),
                _scheme(2, "x + 3", 1.0, "State the quotient"),
            ],
            feedback=[
                _fb(1, "Correctly factorised the dividend and confirmed (2x+1) is a factor."),
                _fb(2, "Correctly identified the remaining factor as the quotient.", improve="Full marks."),
            ],
            topic="polynomial_division", difficulty="medium",
        ),
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Topic: Functions
# ─────────────────────────────────────────────────────────────────────────────

def _functions_examples() -> List[Dict]:
    return [
        _example(
            "If f(x) = 2x^2 - 3x + 1, find f(3)", "function evaluation", 2.0, 2.0,
            steps=[
                _step(1, "f(3) = 2(3)^2 - 3(3) + 1", "correct", 1.0),
                _step(2, "= 18 - 9 + 1 = 10", "correct", 1.0),
            ],
            scheme=[
                _scheme(1, "2(3)^2 - 3(3) + 1", 1.0, "Substitute x=3"),
                _scheme(2, "10", 1.0, "Evaluate"),
            ],
            feedback=[
                _fb(1, "Correctly substituted x=3 into the function."),
                _fb(2, "Correctly evaluated the arithmetic.", improve="Full marks."),
            ],
            topic="functions", difficulty="easy",
        ),
        _example(
            "If f(x) = x^2 + 4, find f(-2)", "function evaluation", 0.0, 1.0,
            steps=[
                _step(1, "f(-2) = (-2)^2 + 4 = -4 + 4 = 0", "incorrect", 0.0,
                      "Sign error: (-2)^2 = 4 (positive), not -4"),
            ],
            scheme=[_scheme(1, "8", 1.0, "Substitute x=-2 and evaluate")],
            feedback=[
                _fb(1, "You correctly substituted x=-2 into the function.",
                    missing="(-2)^2 = (-2)x(-2) = 4, a positive number, not -4. A negative number squared is always positive.",
                    deduction="Full marks lost because squaring a negative number was handled incorrectly.",
                    improve="When squaring a negative number, remember negative times negative = positive. Correct: f(-2) = 4 + 4 = 8."),
            ],
            topic="functions", difficulty="easy",
        ),
        _example(
            "The function f(x) = 3x - 2 has domain {1, 2, 3}. Find the range.", "function evaluation (range)", 2.0, 3.0,
            steps=[
                _step(1, "f(1) = 3(1) - 2 = 1", "correct", 1.0),
                _step(2, "f(2) = 3(2) - 2 = 4", "correct", 1.0),
                _step(3, "f(3) = 3(3) - 2 = 8", "incorrect", 0.0, "Arithmetic slip: 3(3)-2 = 9-2 = 7, not 8"),
            ],
            scheme=[
                _scheme(1, "1", 1.0, "Evaluate f(1)"),
                _scheme(2, "4", 1.0, "Evaluate f(2)"),
                _scheme(3, "7", 1.0, "Evaluate f(3)"),
            ],
            feedback=[
                _fb(1, "Correctly evaluated f(1)."),
                _fb(2, "Correctly evaluated f(2)."),
                _fb(3, "Correct method of substitution.",
                    missing="3(3)-2 = 9-2 = 7, not 8.",
                    deduction="1 mark lost for the arithmetic slip.",
                    improve="Double-check basic arithmetic. Range = {1, 4, 7}."),
            ],
            topic="functions", difficulty="medium",
        ),
        _example(
            "Given f(x) = 5 - 2x, find the value of x for which f(x) = 1", "function (solve for input)", 2.0, 2.0,
            steps=[
                _step(1, "5 - 2x = 1 → -2x = -4", "correct", 1.0),
                _step(2, "x = 2", "correct", 1.0),
            ],
            scheme=[
                _scheme(1, "-2x = -4", 1.0, "Set the function equal to 1 and rearrange"),
                _scheme(2, "x = 2", 1.0, "Solve for x"),
            ],
            feedback=[
                _fb(1, "Correctly rearranged the equation."),
                _fb(2, "Correctly solved for x.", improve="Full marks. Verify: f(2) = 5-4 = 1 ✓."),
            ],
            topic="functions", difficulty="medium",
        ),
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Topic: Function composition
# ─────────────────────────────────────────────────────────────────────────────

def _function_composition_examples() -> List[Dict]:
    return [
        _example(
            "If f(x) = x + 2 and g(x) = 3x, find (f∘g)(x)", "function composition", 2.0, 2.0,
            steps=[
                _step(1, "f(g(x)) = f(3x)", "correct", 1.0),
                _step(2, "= 3x + 2", "correct", 1.0),
            ],
            scheme=[
                _scheme(1, "f(3x)", 1.0, "Substitute g(x) into f"),
                _scheme(2, "3x + 2", 1.0, "Evaluate f at 3x"),
            ],
            feedback=[
                _fb(1, "Correctly substituted g(x) into f — apply g first, then f."),
                _fb(2, "Correctly evaluated.", improve="Full marks."),
            ],
            topic="function_composition", difficulty="easy",
        ),
        _example(
            "If f(x) = x^2 and g(x) = x - 1, find (g∘f)(2)", "function composition", 0.0, 2.0,
            steps=[
                _step(1, "f(g(2)) = f(1) = 1", "incorrect", 0.0,
                      "Wrong order: (g∘f)(x) means g(f(x)) — apply f first, then g"),
            ],
            scheme=[_scheme(1, "g(f(2)) = g(4) = 3", 2.0, "Apply f first, then g")],
            feedback=[
                _fb(1, "You correctly evaluated using one of the two functions first.",
                    missing="(g∘f)(x) means g(f(x)) — apply f first, then g. You computed f(g(2)) instead, which reverses the order. Correct: f(2)=4, then g(4)=4-1=3.",
                    deduction="Full marks lost because the composition was applied in the wrong order.",
                    improve="Read (g∘f)(x) from right to left: f acts first (closest to x), then g acts on that result."),
            ],
            topic="function_composition", difficulty="difficult",
        ),
        _example(
            "If f(x) = 2x - 1, find f(f(3))", "function composition", 2.0, 2.0,
            steps=[
                _step(1, "f(3) = 2(3) - 1 = 5", "correct", 1.0),
                _step(2, "f(5) = 2(5) - 1 = 9", "correct", 1.0),
            ],
            scheme=[
                _scheme(1, "f(3) = 5", 1.0, "Evaluate the inner f(3)"),
                _scheme(2, "f(5) = 9", 1.0, "Evaluate f at the result"),
            ],
            feedback=[
                _fb(1, "Correctly evaluated the inner function first."),
                _fb(2, "Correctly evaluated f applied to the result.", improve="Full marks."),
            ],
            topic="function_composition", difficulty="medium",
        ),
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Topic: Inverse functions
# ─────────────────────────────────────────────────────────────────────────────

def _inverse_functions_examples() -> List[Dict]:
    return [
        _example(
            "Find the inverse of f(x) = 2x + 6", "inverse function", 2.0, 2.0,
            steps=[
                _step(1, "x = 2y + 6", "correct", 1.0),
                _step(2, "y = (x - 6)/2", "correct", 1.0),
            ],
            scheme=[
                _scheme(1, "x = 2y + 6", 1.0, "Swap x and y"),
                _scheme(2, "y = (x-6)/2", 1.0, "Solve for y"),
            ],
            feedback=[
                _fb(1, "Correctly swapped x and y."),
                _fb(2, "Correctly solved for y.",
                    improve="Full marks. Verify: f(f^-1(x)) = 2((x-6)/2)+6 = x-6+6 = x ✓."),
            ],
            topic="inverse_functions", difficulty="medium",
        ),
        _example(
            "Find the inverse of f(x) = (x + 4)/3", "inverse function", 1.0, 2.0,
            steps=[
                _step(1, "x = (y + 4)/3 → 3x = y + 4", "correct", 1.0),
                _step(2, "y = 3x + 4", "incorrect", 0.0, "Sign error: subtract 4 from both sides, giving y=3x-4, not 3x+4"),
            ],
            scheme=[
                _scheme(1, "3x = y + 4", 1.0, "Swap and clear the fraction"),
                _scheme(2, "y = 3x - 4", 1.0, "Solve for y"),
            ],
            feedback=[
                _fb(1, "Correctly cleared the fraction by multiplying both sides by 3."),
                _fb(2, "You correctly began isolating y.",
                    missing="3x=y+4 means y=3x-4 (subtract 4 from both sides), not 3x+4.",
                    deduction="Full marks lost for the sign error.",
                    improve="Verify by checking f(f^-1(x))=x: with the correct inverse, f((3x-4+4)/3)=3x/3=x ✓."),
            ],
            topic="inverse_functions", difficulty="medium",
        ),
        _example(
            "If f(x) = x^3 - 1, find f^-1(7)", "inverse function (solve directly)", 2.0, 2.0,
            steps=[
                _step(1, "x^3 - 1 = 7 → x^3 = 8", "correct", 1.0),
                _step(2, "x = 2", "correct", 1.0),
            ],
            scheme=[
                _scheme(1, "x^3 = 8", 1.0, "Set f(x)=7 and rearrange"),
                _scheme(2, "x = 2", 1.0, "Take the cube root"),
            ],
            feedback=[
                _fb(1, "Correctly rearranged the equation."),
                _fb(2, "Correctly took the cube root of 8.", improve="Full marks. Verify: f(2)=8-1=7 ✓."),
            ],
            topic="inverse_functions", difficulty="difficult",
        ),
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Topic: Straight line equations
# ─────────────────────────────────────────────────────────────────────────────

def _straight_line_examples() -> List[Dict]:
    return [
        _example(
            "Find the equation of the line through (2, 3) with gradient 4", "straight line equation", 2.0, 2.0,
            steps=[
                _step(1, "y - 3 = 4(x - 2)", "correct", 1.0),
                _step(2, "y = 4x - 5", "correct", 1.0),
            ],
            scheme=[
                _scheme(1, "y - 3 = 4(x - 2)", 1.0, "Use point-gradient form"),
                _scheme(2, "y = 4x - 5", 1.0, "Rearrange to y = mx + c"),
            ],
            feedback=[
                _fb(1, "Correctly applied point-gradient form."),
                _fb(2, "Correctly rearranged.", improve="Full marks. Verify: 4(2)-5=3 ✓."),
            ],
            topic="straight_line_equations", difficulty="easy",
        ),
        _example(
            "Find the equation of the line through (1, 5) and (3, 9)", "straight line equation", 0.0, 3.0,
            steps=[
                _step(1, "gradient = (3-1)/(9-5) = 0.5", "incorrect", 0.0,
                      "Gradient formula inverted: should be (y2-y1)/(x2-x1), not (x2-x1)/(y2-y1)"),
                _step(2, "y - 5 = 0.5(x - 1) → y = 0.5x + 4.5", "incorrect", 0.0,
                      "Follows from the incorrect gradient"),
            ],
            scheme=[
                _scheme(1, "gradient = 2", 1.5, "Apply (y2-y1)/(x2-x1) = (9-5)/(3-1)"),
                _scheme(2, "y = 2x + 3", 1.5, "Substitute a point and the gradient"),
            ],
            feedback=[
                _fb(1, "You correctly identified both points and attempted to find the gradient.",
                    missing="Gradient formula is (y2-y1)/(x2-x1), i.e. change in y over change in x — you divided the wrong way round. Correct: (9-5)/(3-1)=4/2=2.",
                    deduction="Full marks lost — the gradient formula was inverted."),
                _fb(2, "Correct method of substituting a point and gradient into y-y1=m(x-x1).",
                    missing="Because the gradient in Step 1 was wrong, this equation is wrong. Using the correct gradient 2: y-5=2(x-1) → y=2x+3.",
                    deduction="Marks lost because this follows from the earlier gradient error.",
                    improve="Always remember gradient = rise/run = (change in y)/(change in x). Verify with both points: at x=1, y=5 ✓; at x=3, y=9 ✓."),
            ],
            topic="straight_line_equations", difficulty="medium",
        ),
        _example(
            "A line has equation 2x + 3y = 12. Find its gradient.", "straight line equation (rearranging)", 1.5, 2.0,
            steps=[
                _step(1, "3y = -2x + 12", "correct", 1.0),
                _step(2, "y = (-2/3)x + 4, gradient = 2/3", "partial", 0.5,
                      "Sign dropped — gradient should be -2/3, not 2/3"),
            ],
            scheme=[
                _scheme(1, "3y = -2x + 12", 1.0, "Rearrange into y = mx + c form"),
                _scheme(2, "gradient = -2/3", 1.0, "Read off the gradient, keeping the sign"),
            ],
            feedback=[
                _fb(1, "You correctly rearranged the equation."),
                _fb(2, "You correctly divided through by 3.",
                    missing="The x-coefficient is -2, so after dividing by 3 the gradient is -2/3, not 2/3 — the negative sign must be kept.",
                    deduction="0.5 mark deducted for dropping the negative sign on the gradient.",
                    improve="Carry negative signs through every step of an algebraic rearrangement."),
            ],
            topic="straight_line_equations", difficulty="medium",
        ),
        _example(
            "Find where the line y = 3x - 6 crosses the x-axis", "straight line equation (intercept)", 2.0, 2.0,
            steps=[
                _step(1, "0 = 3x - 6 → 3x = 6", "correct", 1.0),
                _step(2, "x = 2, so the line crosses at (2, 0)", "correct", 1.0),
            ],
            scheme=[
                _scheme(1, "3x = 6", 1.0, "Set y=0"),
                _scheme(2, "(2, 0)", 1.0, "Solve for x"),
            ],
            feedback=[
                _fb(1, "Correctly set y=0 to find the x-intercept."),
                _fb(2, "Correctly solved for x.", improve="Full marks."),
            ],
            topic="straight_line_equations", difficulty="easy",
        ),
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Topic: Gradient and intercept
# ─────────────────────────────────────────────────────────────────────────────

def _gradient_intercept_examples() -> List[Dict]:
    return [
        _example(
            "State the gradient and y-intercept of y = -4x + 7", "gradient and intercept", 1.0, 1.0,
            steps=[_step(1, "gradient = -4, y-intercept = 7", "correct", 1.0)],
            scheme=[_scheme(1, "gradient = -4, y-intercept = 7", 1.0, "Read directly from y = mx + c form")],
            feedback=[_fb(1, "Correctly identified both values directly from y=mx+c form.", improve="Full marks.")],
            topic="gradient_intercept", difficulty="easy",
        ),
        _example(
            "Find the equation of a line parallel to y = 2x + 1 passing through (0, 5)",
            "gradient and intercept (parallel lines)", 0.0, 2.0,
            steps=[
                _step(1, "y = 2x + 1", "incorrect", 0.0,
                      "Misunderstood 'parallel': kept the original equation instead of finding the new line's own intercept"),
            ],
            scheme=[_scheme(1, "y = 2x + 5", 2.0, "Keep the gradient (2), but use the new point for the intercept")],
            feedback=[
                _fb(1, "You correctly identified the gradient of the given line as 2.",
                    missing="Parallel lines share the same gradient (2) but are different lines with their own y-intercept. Since the new line passes through (0,5), its y-intercept is 5, giving y=2x+5, not the original equation unchanged.",
                    deduction="Full marks lost because the new line's own intercept was never calculated.",
                    improve="For a parallel line, keep the gradient the same but substitute the new point into y=mx+c to find its own c."),
            ],
            topic="gradient_intercept", difficulty="medium",
        ),
        _example(
            "Find the gradient of a line perpendicular to y = (1/2)x - 3", "gradient and intercept (perpendicular lines)", 1.0, 1.0,
            steps=[_step(1, "perpendicular gradient = -1/(1/2) = -2", "correct", 1.0)],
            scheme=[_scheme(1, "-2", 1.0, "Take the negative reciprocal of the original gradient")],
            feedback=[_fb(1, "Correctly took the negative reciprocal of 1/2.", improve="Full marks.")],
            topic="gradient_intercept", difficulty="medium",
        ),
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Topic: Sequences — nth term
# ─────────────────────────────────────────────────────────────────────────────

def _sequences_nth_term_examples() -> List[Dict]:
    return [
        _example(
            "Find the nth term of the sequence 5, 8, 11, 14, ...", "sequences (nth term)", 1.0, 1.0,
            steps=[
                _step(1, "common difference d = 3", "correct", 0.5),
                _step(2, "nth term = 3n + 2", "correct", 0.5),
            ],
            scheme=[
                _scheme(1, "d = 3", 0.5, "Find the common difference"),
                _scheme(2, "3n + 2", 0.5, "Apply nth term = a + (n-1)d, simplified"),
            ],
            feedback=[
                _fb(1, "Correctly identified the common difference."),
                _fb(2, "Correctly formed the nth term expression.", improve="Full marks. Check: n=1 gives 5 ✓."),
            ],
            topic="sequences_nth_term", difficulty="easy",
        ),
        _example(
            "Find the nth term of the sequence 2, 6, 12, 20, 30, ...", "sequences (nth term, quadratic)", 0.0, 2.0,
            steps=[
                _step(1, "differences: 4, 6, 8, 10 — treated as arithmetic with d=4", "incorrect", 0.0,
                      "The first differences are not constant, so this is not a simple arithmetic sequence"),
            ],
            scheme=[_scheme(1, "n^2 + n", 2.0, "Second differences are constant (=2), so the sequence is quadratic")],
            feedback=[
                _fb(1, "You correctly found the first differences between terms.",
                    missing="Since the first differences (4,6,8,10) are not constant, this is not an arithmetic sequence. Check the SECOND differences (6-4=2, 8-6=2, 10-8=2) — since these are constant, the sequence is quadratic: nth term = n^2+n.",
                    deduction="Full marks lost for treating a quadratic sequence as arithmetic.",
                    improve="Always check if the first differences are constant. If not, check the second differences — a constant second difference tells you the sequence is quadratic."),
            ],
            topic="sequences_nth_term", difficulty="difficult",
        ),
        _example(
            "Find the 10th term of the sequence with nth term 5n - 3", "sequences (nth term, evaluation)", 1.0, 1.0,
            steps=[_step(1, "5(10) - 3 = 47", "correct", 1.0)],
            scheme=[_scheme(1, "47", 1.0, "Substitute n=10")],
            feedback=[_fb(1, "Correctly substituted and evaluated.", improve="Full marks.")],
            topic="sequences_nth_term", difficulty="easy",
        ),
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Topic: Algebraic word problems
# ─────────────────────────────────────────────────────────────────────────────

def _algebraic_word_problems_examples() -> List[Dict]:
    return [
        _example(
            "The sum of a number and twice that number is 27. Find the number.",
            "algebraic word problem", 2.0, 2.0,
            steps=[
                _step(1, "x + 2x = 27 → 3x = 27", "correct", 1.0),
                _step(2, "x = 9", "correct", 1.0),
            ],
            scheme=[
                _scheme(1, "3x = 27", 1.0, "Translate the words into an equation"),
                _scheme(2, "x = 9", 1.0, "Solve for x"),
            ],
            feedback=[
                _fb(1, "Correctly translated the problem into an equation."),
                _fb(2, "Correctly solved for x.", improve="Full marks. Verify: 9+18=27 ✓."),
            ],
            topic="algebraic_word_problems", difficulty="easy",
        ),
        _example(
            "A rectangle's length is 3 cm more than its width. If the perimeter is 26 cm, find the width.",
            "algebraic word problem", 0.0, 3.0,
            steps=[
                _step(1, "w + (w + 3) = 26", "incorrect", 0.0,
                      "Perimeter of a rectangle is 2x(length+width), not just length+width — the factor of 2 was omitted"),
                _step(2, "2w + 3 = 26 → w = 11.5", "incorrect", 0.0, "Follows from the missing factor of 2"),
            ],
            scheme=[
                _scheme(1, "2(w + (w+3)) = 26", 1.5, "Set up the perimeter equation correctly"),
                _scheme(2, "w = 5", 1.5, "Solve for w"),
            ],
            feedback=[
                _fb(1, "You correctly set up an expression for length in terms of width (w+3).",
                    missing="The perimeter formula is P=2(length+width), not just length+width. It should be 2(w+(w+3))=26.",
                    deduction="Full marks lost because the factor of 2 for perimeter was omitted."),
                _fb(2, "Your algebra correctly followed from your Step 1 equation.",
                    missing="Because Step 1 was missing the factor of 2, this answer is wrong. Solving 2(2w+3)=26 correctly gives w=5.",
                    deduction="Marks lost because this follows from the earlier setup error.",
                    improve="Always write the full standard formula first (e.g. P=2(l+w)) before substituting expressions — this avoids missing factors."),
            ],
            topic="algebraic_word_problems", difficulty="medium",
        ),
        _example(
            "Three consecutive integers sum to 72. Find the largest integer.",
            "algebraic word problem", 2.5, 3.0,
            steps=[
                _step(1, "n + (n+1) + (n+2) = 72 → 3n + 3 = 72", "correct", 1.0),
                _step(2, "n = 23", "correct", 1.0),
                _step(3, "largest integer = 23", "partial", 0.5,
                      "Answered with the smallest integer (n) instead of the largest (n+2 = 25)"),
            ],
            scheme=[
                _scheme(1, "3n + 3 = 72", 1.0, "Translate into an equation"),
                _scheme(2, "n = 23", 1.0, "Solve for n"),
                _scheme(3, "25", 1.0, "State the largest integer (n+2)"),
            ],
            feedback=[
                _fb(1, "Correctly translated the problem into an equation."),
                _fb(2, "Correctly solved for n."),
                _fb(3, "Your algebra correctly solved for n=23.",
                    missing="n=23 is the SMALLEST of the three consecutive integers, but the question asks for the LARGEST, which is n+2 = 25.",
                    deduction="0.5 mark deducted for answering with the wrong integer in the sequence — the calculation was right, but the wrong one was reported as the answer.",
                    improve="Always re-read the question after solving to check exactly which value it's asking for — smallest, middle, or largest."),
            ],
            topic="algebraic_word_problems", difficulty="medium",
        ),
        _example(
            "A number increased by 20% gives 60. Find the original number.",
            "algebraic word problem", 2.0, 2.0,
            steps=[
                _step(1, "1.2x = 60", "correct", 1.0),
                _step(2, "x = 50", "correct", 1.0),
            ],
            scheme=[
                _scheme(1, "1.2x = 60", 1.0, "Translate '20% increase' into a multiplier"),
                _scheme(2, "x = 50", 1.0, "Solve for x"),
            ],
            feedback=[
                _fb(1, "Correctly translated a 20% increase as multiplying by 1.2."),
                _fb(2, "Correctly solved for x.", improve="Full marks. Verify: 50 x 1.2 = 60 ✓."),
            ],
            topic="algebraic_word_problems", difficulty="medium",
        ),
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Topic: Ratio and proportion
# ─────────────────────────────────────────────────────────────────────────────

def _ratio_proportion_examples() -> List[Dict]:
    return [
        _example(
            "Divide 60 in the ratio 2:3", "ratio and proportion", 2.0, 2.0,
            steps=[
                _step(1, "total parts = 2+3 = 5, each part = 60/5 = 12", "correct", 1.0),
                _step(2, "24 : 36", "correct", 1.0),
            ],
            scheme=[
                _scheme(1, "each part = 12", 1.0, "Find the value of one part"),
                _scheme(2, "24 : 36", 1.0, "Multiply each ratio number by the part value"),
            ],
            feedback=[
                _fb(1, "Correctly found the value of one part."),
                _fb(2, "Correctly scaled up both parts.", improve="Full marks. Verify: 24+36=60 ✓."),
            ],
            topic="ratio_proportion", difficulty="easy",
        ),
        _example(
            "If a:b = 3:4 and b:c = 2:5, find a:b:c", "ratio and proportion (combining ratios)", 0.0, 2.0,
            steps=[
                _step(1, "a:b:c = 3:4:5", "incorrect", 0.0,
                      "The shared term b must match in both ratios before combining — b=4 in the first ratio but b=2 in the second"),
            ],
            scheme=[_scheme(1, "3:4:10", 2.0, "Scale b:c=2:5 by 2 so b=4 matches, giving b:c=4:10, then combine")],
            feedback=[
                _fb(1, "You correctly used a=3 and b=4 from the first ratio.",
                    missing="The b value must match in both ratios before combining. In b:c=2:5, b=2, not 4. Scale that ratio by 2 so b becomes 4: b:c=4:10. Now both ratios agree on b=4, giving a:b:c=3:4:10.",
                    deduction="Full marks lost because c was taken directly from the unscaled second ratio (5) instead of the correctly scaled value (10).",
                    improve="To combine two ratios sharing a term, find the LCM of that term's two given values, scale each ratio accordingly, then combine."),
            ],
            topic="ratio_proportion", difficulty="difficult",
        ),
        _example(
            "y is directly proportional to x. When x=4, y=20. Find y when x=7.",
            "ratio and proportion (direct variation)", 2.0, 2.0,
            steps=[
                _step(1, "y = kx → 20 = 4k → k = 5", "correct", 1.0),
                _step(2, "y = 5(7) = 35", "correct", 1.0),
            ],
            scheme=[
                _scheme(1, "k = 5", 1.0, "Find the constant of proportionality"),
                _scheme(2, "y = 35", 1.0, "Substitute x=7"),
            ],
            feedback=[
                _fb(1, "Correctly found the constant of proportionality."),
                _fb(2, "Correctly substituted to find y.", improve="Full marks."),
            ],
            topic="ratio_proportion", difficulty="medium",
        ),
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Topic: Rearranging formulas
# ─────────────────────────────────────────────────────────────────────────────

def _rearranging_formulas_examples() -> List[Dict]:
    return [
        _example(
            "Make x the subject of y = 3x + 7", "rearranging formulas", 2.0, 2.0,
            steps=[
                _step(1, "y - 7 = 3x", "correct", 1.0),
                _step(2, "x = (y - 7)/3", "correct", 1.0),
            ],
            scheme=[
                _scheme(1, "y - 7 = 3x", 1.0, "Subtract 7 from both sides"),
                _scheme(2, "x = (y-7)/3", 1.0, "Divide both sides by 3"),
            ],
            feedback=[
                _fb(1, "Correctly subtracted 7 from both sides."),
                _fb(2, "Correctly divided by 3.", improve="Full marks."),
            ],
            topic="rearranging_formulas", difficulty="easy",
        ),
        _example(
            "Make r the subject of A = πr^2", "rearranging formulas", 1.0, 2.0,
            steps=[
                _step(1, "r^2 = A/π", "correct", 1.0),
                _step(2, "r = A/π", "incorrect", 0.0, "Forgot to take the square root — r^2=A/π means r=√(A/π)"),
            ],
            scheme=[
                _scheme(1, "r^2 = A/π", 1.0, "Divide both sides by π"),
                _scheme(2, "r = √(A/π)", 1.0, "Take the square root of both sides"),
            ],
            feedback=[
                _fb(1, "Correctly isolated r^2 by dividing both sides by π."),
                _fb(2, "You correctly isolated r^2.",
                    missing="To undo the square on r, take the square root of both sides: r=√(A/π), not just A/π.",
                    deduction="Full marks lost because the square root step was skipped.",
                    improve="Whenever a variable is squared, the inverse operation to isolate it is a square root — never skip it."),
            ],
            topic="rearranging_formulas", difficulty="medium",
        ),
        _example(
            "Make h the subject of V = (1/3)πr^2h", "rearranging formulas", 2.0, 2.0,
            steps=[
                _step(1, "3V = πr^2h", "correct", 1.0),
                _step(2, "h = 3V/(πr^2)", "correct", 1.0),
            ],
            scheme=[
                _scheme(1, "3V = πr^2h", 1.0, "Multiply both sides by 3"),
                _scheme(2, "h = 3V/(πr^2)", 1.0, "Divide both sides by πr^2"),
            ],
            feedback=[
                _fb(1, "Correctly multiplied both sides by 3."),
                _fb(2, "Correctly divided both sides by πr^2.", improve="Full marks."),
            ],
            topic="rearranging_formulas", difficulty="medium",
        ),
        _example(
            "Make x the subject of y = (x + 3)/(x - 2)", "rearranging formulas", 3.0, 4.0,
            steps=[
                _step(1, "y(x - 2) = x + 3 → yx - 2y = x + 3", "correct", 1.5),
                _step(2, "yx - x = 3 + 2y", "correct", 1.0),
                _step(3, "x(y - 1) = 3 + 2y", "partial", 0.5,
                      "Factoring step started but never divided by (y-1) to fully isolate x"),
            ],
            scheme=[
                _scheme(1, "yx - 2y = x + 3", 1.5, "Clear the fraction and expand"),
                _scheme(2, "yx - x = 3 + 2y", 1.0, "Collect x terms on one side"),
                _scheme(3, "x = (3 + 2y)/(y - 1)", 1.5, "Factor out x, then divide by (y-1)"),
            ],
            feedback=[
                _fb(1, "Correctly cleared the fraction and expanded."),
                _fb(2, "Correctly collected the x terms onto one side."),
                _fb(3, "You correctly factored x out of both x-terms, getting x(y-1)=3+2y.",
                    missing="The final step is to divide both sides by (y-1) to fully isolate x: x=(3+2y)/(y-1).",
                    deduction="0.5 mark deducted because the formula was not fully solved for x — dividing by (y-1) was never carried out.",
                    improve="When x appears on both sides, collect all x-terms on one side, factor x out, then divide by whatever remains to isolate it completely."),
            ],
            topic="rearranging_formulas", difficulty="difficult",
        ),
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Topic: Absolute value equations
# ─────────────────────────────────────────────────────────────────────────────

def _absolute_value_examples() -> List[Dict]:
    return [
        _example(
            "Solve |x - 4| = 7", "absolute value equation", 2.0, 2.0,
            steps=[
                _step(1, "x - 4 = 7 or x - 4 = -7", "correct", 1.0),
                _step(2, "x = 11 or x = -3", "correct", 1.0),
            ],
            scheme=[
                _scheme(1, "x-4=7 or x-4=-7", 1.0, "Split into the positive and negative case"),
                _scheme(2, "x = 11 or x = -3", 1.0, "Solve each case"),
            ],
            feedback=[
                _fb(1, "Correctly split into both cases."),
                _fb(2, "Correctly solved both cases.", improve="Full marks. Verify: |11-4|=7 ✓ and |-3-4|=7 ✓."),
            ],
            topic="absolute_value_equations", difficulty="medium",
        ),
        _example(
            "Solve |2x + 1| = 9", "absolute value equation", 1.0, 2.0,
            steps=[
                _step(1, "2x + 1 = 9 → x = 4", "partial", 1.0,
                      "Only the positive case was considered — the negative case 2x+1=-9 was never solved"),
            ],
            scheme=[_scheme(1, "x = 4 or x = -5", 2.0, "Split into both the positive and negative case")],
            feedback=[
                _fb(1, "x=4 is one correct solution, obtained correctly from the positive case 2x+1=9.",
                    missing="|2x+1|=9 also requires the negative case: 2x+1=-9, giving x=-5. Both solutions must be stated.",
                    deduction="1 mark deducted because only one of the two required cases was considered.",
                    improve="Whenever solving |expression|=k (k>0), always split into TWO equations: expression=k AND expression=-k."),
            ],
            topic="absolute_value_equations", difficulty="medium",
        ),
        _example(
            "Solve |x + 5| = -3", "absolute value equation (no solution)", 0.0, 2.0,
            steps=[
                _step(1, "x + 5 = -3 or x + 5 = 3 → x = -8 or x = -2", "incorrect", 0.0,
                      "An absolute value can never equal a negative number — this equation has no solution"),
            ],
            scheme=[_scheme(1, "No solution", 2.0, "Recognise that an absolute value can never equal a negative number")],
            feedback=[
                _fb(1, "Your algebraic technique for solving absolute value equations (splitting into two cases) is generally correct.",
                    missing="Before splitting into cases, always check the right-hand side: |expression|=k only has solutions when k is greater than or equal to 0. Since -3 is negative, this equation has NO SOLUTION — there is no valid case to split into.",
                    deduction="Full marks lost because a nonexistent solution was stated as if it were valid.",
                    improve="Always check the sign of the right-hand side before solving |expression|=k. If it is negative, immediately conclude 'no solution' without further algebra."),
            ],
            topic="absolute_value_equations", difficulty="difficult",
        ),
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Topic: Exponent equations
# ─────────────────────────────────────────────────────────────────────────────

def _exponent_equations_examples() -> List[Dict]:
    return [
        _example(
            "Solve 2^x = 32", "exponent equation", 2.0, 2.0,
            steps=[
                _step(1, "32 = 2^5", "correct", 1.0),
                _step(2, "x = 5", "correct", 1.0),
            ],
            scheme=[
                _scheme(1, "32 = 2^5", 1.0, "Rewrite 32 as a power of 2"),
                _scheme(2, "x = 5", 1.0, "Equate the exponents"),
            ],
            feedback=[
                _fb(1, "Correctly rewrote 32 as a power of 2."),
                _fb(2, "Correctly equated the exponents.", improve="Full marks."),
            ],
            topic="exponent_equations", difficulty="easy",
        ),
        _example(
            "Solve 3^(x+1) = 27", "exponent equation", 0.0, 2.0,
            steps=[
                _step(1, "x + 1 = 27 → x = 26", "incorrect", 0.0,
                      "27 must first be rewritten as a power of 3 before equating exponents"),
            ],
            scheme=[_scheme(1, "x = 2", 2.0, "Rewrite 27=3^3, then equate exponents: x+1=3")],
            feedback=[
                _fb(1, "You correctly recognised this is an exponential equation with matching bases.",
                    missing="27 must first be rewritten as a power of 3: 27=3^3. Only then can you equate the exponents: x+1=3, giving x=2 — not x+1=27.",
                    deduction="Full marks lost because the right-hand side was not converted to the same base before equating exponents.",
                    improve="To solve a^m = a^n, first make sure BOTH sides are written as powers of the same base a, then set the exponents equal: m=n."),
            ],
            topic="exponent_equations", difficulty="medium",
        ),
        _example(
            "Solve 4^x = 8", "exponent equation (different bases)", 2.0, 2.0,
            steps=[
                _step(1, "4^x = 2^(2x), 8 = 2^3", "correct", 1.0),
                _step(2, "2x = 3 → x = 1.5", "correct", 1.0),
            ],
            scheme=[
                _scheme(1, "2^(2x) = 2^3", 1.0, "Rewrite both sides with base 2"),
                _scheme(2, "x = 1.5", 1.0, "Equate exponents and solve"),
            ],
            feedback=[
                _fb(1, "Correctly rewrote both sides using base 2."),
                _fb(2, "Correctly equated exponents and solved.",
                    improve="Full marks. Verify: 4^1.5 = 4 x sqrt(4) = 4 x 2 = 8 ✓."),
            ],
            topic="exponent_equations", difficulty="difficult",
        ),
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Top-ups: thin topics (equations_with_brackets, equations_with_fractions,
# simultaneous_substitution, negative_fractional_indices, rationalising_denominators
# each previously had only 1 example) plus a few partial/difficult additions to
# existing topics to correct the correct/partial/incorrect balance.
# ─────────────────────────────────────────────────────────────────────────────

def _equations_with_brackets_topup_examples() -> List[Dict]:
    return [
        _example(
            "Solve 3(x + 2) = 21", "linear equation with brackets", 2.0, 2.0,
            steps=[
                _step(1, "3x + 6 = 21 → 3x = 15", "correct", 1.0),
                _step(2, "x = 5", "correct", 1.0),
            ],
            scheme=[
                _scheme(1, "3x = 15", 1.0, "Distribute the 3, then subtract 6"),
                _scheme(2, "x = 5", 1.0, "Divide both sides by 3"),
            ],
            feedback=[
                _fb(1, "Correctly distributed the 3 and isolated the x term."),
                _fb(2, "Correctly solved for x.", improve="Full marks. Verify: 3(5+2)=21 ✓."),
            ],
            topic="equations_with_brackets", difficulty="easy",
        ),
        _example(
            "Solve 4(x + 3) = 2(x + 9)", "linear equation with brackets", 0.0, 2.0,
            steps=[
                _step(1, "4x + 12 = 2x + 9", "incorrect", 0.0,
                      "RHS not fully distributed: 2(x+9)=2x+18, not 2x+9 — both terms must be multiplied by 2"),
                _step(2, "2x = -3 → x = -1.5", "incorrect", 0.0, "Follows from the distribution error"),
            ],
            scheme=[
                _scheme(1, "4x + 12 = 2x + 18", 1.0, "Distribute both sides fully"),
                _scheme(2, "x = 3", 1.0, "Solve for x"),
            ],
            feedback=[
                _fb(1, "You correctly distributed the 4 on the left-hand side.",
                    missing="On the right-hand side, 2(x+9) means BOTH the x and the 9 must be multiplied by 2: 2x+18, not 2x+9.",
                    deduction="Full marks lost because the right-hand bracket was not fully distributed."),
                _fb(2, "Your algebra correctly followed from your Step 1 equation.",
                    missing="Because Step 1 was wrong, this answer is wrong. Solving 4x+12=2x+18 correctly gives 2x=6, x=3.",
                    deduction="Marks lost because this follows from the earlier distribution error.",
                    improve="Always distribute a bracket to EVERY term inside it, on both sides of the equation."),
            ],
            topic="equations_with_brackets", difficulty="medium",
        ),
        _example(
            "Solve 2(3x - 1) + 4 = 20", "linear equation with brackets", 2.5, 3.0,
            steps=[
                _step(1, "6x - 2 + 4 = 20 → 6x + 2 = 20", "correct", 1.0),
                _step(2, "6x = 18", "correct", 1.0),
                _step(3, "x = 18/6", "partial", 0.5, "Not simplified — should be x = 3"),
            ],
            scheme=[
                _scheme(1, "6x + 2 = 20", 1.0, "Distribute the 2, then combine constants"),
                _scheme(2, "6x = 18", 1.0, "Subtract 2 from both sides"),
                _scheme(3, "x = 3", 1.0, "Divide both sides by 6 and simplify"),
            ],
            feedback=[
                _fb(1, "Correctly distributed the 2 and combined the constants."),
                _fb(2, "Correctly isolated the x term."),
                _fb(3, "Correct method of dividing by 6.",
                    missing="18/6 must be evaluated to a final number: x=3.",
                    deduction="0.5 mark deducted because the fraction was left unsimplified.",
                    improve="Always finish arithmetic to a final simplified number."),
            ],
            topic="equations_with_brackets", difficulty="easy",
        ),
    ]


def _equations_with_fractions_topup_examples() -> List[Dict]:
    return [
        _example(
            "Solve x/4 + 3 = 7", "linear equation with fractions", 2.0, 2.0,
            steps=[
                _step(1, "x/4 = 4", "correct", 1.0),
                _step(2, "x = 16", "correct", 1.0),
            ],
            scheme=[
                _scheme(1, "x/4 = 4", 1.0, "Subtract 3 from both sides"),
                _scheme(2, "x = 16", 1.0, "Multiply both sides by 4"),
            ],
            feedback=[
                _fb(1, "Correctly isolated the fraction term."),
                _fb(2, "Correctly multiplied both sides by 4.", improve="Full marks."),
            ],
            topic="equations_with_fractions", difficulty="easy",
        ),
        _example(
            "Solve (x - 2)/5 = 3", "linear equation with fractions", 1.5, 2.0,
            steps=[
                _step(1, "x - 2 = 15", "correct", 1.0),
                _step(2, "x = 2 + 15", "partial", 0.5, "Not evaluated to a final number — should be x = 17"),
            ],
            scheme=[
                _scheme(1, "x - 2 = 15", 1.0, "Multiply both sides by 5"),
                _scheme(2, "x = 17", 1.0, "Add 2 to both sides and evaluate"),
            ],
            feedback=[
                _fb(1, "Correctly cleared the fraction by multiplying both sides by 5."),
                _fb(2, "Correct method of adding 2 to both sides.",
                    missing="2 + 15 must be evaluated to a final number: x=17.",
                    deduction="0.5 mark deducted because the answer was left unevaluated.",
                    improve="Always finish arithmetic to a final simplified number."),
            ],
            topic="equations_with_fractions", difficulty="easy",
        ),
        _example(
            "Solve x/3 + x/6 = 3", "linear equation with fractions", 0.0, 3.0,
            steps=[
                _step(1, "x/3 + x/6 = 2x/9 = 3", "incorrect", 0.0,
                      "Cannot add fractions by adding denominators directly — a common denominator is required"),
                _step(2, "2x = 27 → x = 13.5", "incorrect", 0.0, "Follows from the invalid fraction addition"),
            ],
            scheme=[
                _scheme(1, "2x/6 + x/6 = 3x/6", 1.5, "Use the common denominator 6"),
                _scheme(2, "x = 6", 1.5, "Solve for x"),
            ],
            feedback=[
                _fb(1, "You attempted to combine the two fractions into one.",
                    missing="Fractions cannot be added by adding their denominators. Convert both to a common denominator of 6 first: x/3=2x/6, so x/3+x/6=2x/6+x/6=3x/6.",
                    deduction="Full marks lost because the fraction addition rule was applied incorrectly."),
                _fb(2, "Your algebra correctly followed from your Step 1 expression.",
                    missing="Because Step 1 was invalid, this answer is wrong. Solving 3x/6=3 correctly gives x=6.",
                    deduction="Marks lost because this follows from the earlier error.",
                    improve="To add fractions with different denominators, always find a common denominator first."),
            ],
            topic="equations_with_fractions", difficulty="difficult",
        ),
    ]


def _simultaneous_substitution_topup_examples() -> List[Dict]:
    return [
        _example(
            "Solve: y = 2x - 1, 3x + y = 14", "substitution", 2.0, 2.0,
            steps=[
                _step(1, "3x + 2x - 1 = 14 → 5x = 15", "correct", 1.0),
                _step(2, "x = 3, y = 5", "correct", 1.0),
            ],
            scheme=[
                _scheme(1, "5x = 15", 1.0, "Substitute y = 2x-1"),
                _scheme(2, "x = 3, y = 5", 1.0, "Solve for x, then find y"),
            ],
            feedback=[
                _fb(1, "Correctly substituted y=2x-1 into the second equation."),
                _fb(2, "Correctly solved for x and y.", improve="Full marks. Verify: 3(3)+5=14 ✓."),
            ],
            topic="simultaneous_substitution", difficulty="easy",
        ),
        _example(
            "Solve: x = y + 3, 2x - y = 11", "substitution", 0.0, 3.0,
            steps=[
                _step(1, "2y + 3 - y = 11", "incorrect", 0.0,
                      "Distribution error: 2(y+3) = 2y+6, not 2y+3"),
                _step(2, "y = 8, x = 11", "incorrect", 0.0, "Follows from the distribution error"),
            ],
            scheme=[
                _scheme(1, "2y + 6 - y = 11", 1.5, "Substitute x = y+3 and distribute fully"),
                _scheme(2, "y = 5, x = 8", 1.5, "Solve for y, then find x"),
            ],
            feedback=[
                _fb(1, "You correctly set up the substitution x=y+3 into the second equation.",
                    missing="2(y+3) means both terms must be multiplied by 2: 2y+6, not 2y+3.",
                    deduction="Full marks lost because the bracket was not fully distributed."),
                _fb(2, "Your algebra correctly followed from your Step 1 equation.",
                    missing="Because Step 1 was wrong, this answer is wrong. Solving 2y+6-y=11 correctly gives y=5, x=8.",
                    deduction="Marks lost because this follows from the earlier distribution error.",
                    improve="Always distribute a substituted expression to every term when it's multiplied by a coefficient."),
            ],
            topic="simultaneous_substitution", difficulty="medium",
        ),
        _example(
            "Solve: 2x + y = 9, y = x - 3", "substitution", 2.0, 3.0,
            steps=[
                _step(1, "2x + x - 3 = 9 → 3x = 12", "correct", 1.0),
                _step(2, "x = 4", "correct", 1.0),
                _step(3, "y = 4 - 3 = 2", "incorrect", 0.0, "Arithmetic slip: 4-3=1, not 2"),
            ],
            scheme=[
                _scheme(1, "3x = 12", 1.0, "Substitute y = x-3"),
                _scheme(2, "x = 4", 1.0, "Solve for x"),
                _scheme(3, "y = 1", 1.0, "Substitute back to find y"),
            ],
            feedback=[
                _fb(1, "Correctly substituted y=x-3 into the first equation."),
                _fb(2, "Correctly solved for x."),
                _fb(3, "Correct method of substituting back to find y.",
                    missing="4 - 3 = 1, not 2 — this looks like an arithmetic slip.",
                    deduction="Marks lost for the incorrect subtraction despite the correct method.",
                    improve="Double-check basic subtraction. Verify: 2(4)+1=9 ✓."),
            ],
            topic="simultaneous_substitution", difficulty="medium",
        ),
    ]


def _negative_fractional_indices_topup_examples() -> List[Dict]:
    return [
        _example(
            "Simplify x^-3", "index laws (negative indices)", 1.0, 1.0,
            steps=[_step(1, "1/x^3", "correct", 1.0)],
            scheme=[_scheme(1, "1/x^3", 1.0, "A negative exponent means the reciprocal")],
            feedback=[_fb(1, "Correctly applied a^-n = 1/a^n.", improve="Full marks.")],
            topic="negative_fractional_indices", difficulty="easy",
        ),
        _example(
            "Evaluate 16^(1/2)", "index laws (fractional indices)", 1.0, 1.0,
            steps=[_step(1, "√16 = 4", "correct", 1.0)],
            scheme=[_scheme(1, "4", 1.0, "A power of 1/2 means the square root")],
            feedback=[_fb(1, "Correctly recognised a^(1/2) as the square root.", improve="Full marks.")],
            topic="negative_fractional_indices", difficulty="easy",
        ),
        _example(
            "Evaluate 27^(-2/3)", "index laws (negative fractional indices)", 1.0, 2.0,
            steps=[
                _step(1, "27^(1/3) = 3", "correct", 1.0),
                _step(2, "3^2 = 9", "incorrect", 0.0,
                      "The negative sign on the exponent was ignored — a negative exponent means take the reciprocal"),
            ],
            scheme=[
                _scheme(1, "27^(1/3) = 3", 1.0, "Take the cube root first (denominator of the fractional exponent)"),
                _scheme(2, "1/9", 1.0, "Apply the negative exponent as a reciprocal of 3^2"),
            ],
            feedback=[
                _fb(1, "Correctly took the cube root of 27."),
                _fb(2, "You correctly squared the cube root.",
                    missing="The exponent is -2/3, and the negative sign means take the RECIPROCAL: 3^-2 = 1/3^2 = 1/9, not 3^2 = 9.",
                    deduction="Full marks lost because the negative sign on the exponent was dropped.",
                    improve="Handle a negative fractional exponent in three steps: take the root (denominator), raise to the power (numerator), then reciprocate (the negative sign)."),
            ],
            topic="negative_fractional_indices", difficulty="difficult",
        ),
    ]


def _rationalising_denominators_topup_examples() -> List[Dict]:
    return [
        _example(
            "Rationalise 5/√2", "surds (rationalisation)", 1.0, 1.0,
            steps=[_step(1, "5√2/2", "correct", 1.0)],
            scheme=[_scheme(1, "5√2/2", 1.0, "Multiply top and bottom by √2")],
            feedback=[_fb(1, "Correctly multiplied by √2/√2 and simplified the denominator.", improve="Full marks.")],
            topic="rationalising_denominators", difficulty="easy",
        ),
        _example(
            "Rationalise 3/(2√5)", "surds (rationalisation)", 0.0, 2.0,
            steps=[
                _step(1, "3/(2√5) × √5/√5 = 3/10", "incorrect", 0.0,
                      "Numerator was not multiplied by √5 — only the denominator was rationalised"),
            ],
            scheme=[_scheme(1, "3√5/10", 2.0, "Multiply BOTH the numerator and denominator by √5")],
            feedback=[
                _fb(1, "You correctly multiplied the denominator by √5, clearing the surd there.",
                    missing="When rationalising, the numerator must ALSO be multiplied by √5: 3×√5 = 3√5. The final answer should be 3√5/10, not 3/10.",
                    deduction="Full marks lost because the numerator was left unmultiplied.",
                    improve="Rationalising means multiplying the WHOLE fraction by (√a/√a) — apply it to both the numerator and the denominator."),
            ],
            topic="rationalising_denominators", difficulty="medium",
        ),
        _example(
            "Rationalise 4/(√7 - 1)", "surds (rationalisation with conjugate)", 2.0, 2.0,
            steps=[
                _step(1, "4(√7 + 1) / ((√7 - 1)(√7 + 1))", "correct", 1.0),
                _step(2, "= 4(√7 + 1)/6 = 2(√7 + 1)/3", "correct", 1.0),
            ],
            scheme=[
                _scheme(1, "4(√7+1)/((√7-1)(√7+1))", 1.0, "Multiply top and bottom by the conjugate √7+1"),
                _scheme(2, "2(√7+1)/3", 1.0, "Simplify: (√7)^2-1^2=6, then reduce the fraction"),
            ],
            feedback=[
                _fb(1, "Correctly multiplied by the conjugate of the denominator."),
                _fb(2, "Correctly simplified using the difference-of-squares identity and reduced the fraction.",
                    improve="Full marks."),
            ],
            topic="rationalising_denominators", difficulty="difficult",
        ),
    ]


def _expanding_brackets_topup_examples() -> List[Dict]:
    return [
        _example(
            "Expand (2x + 3)(x - 5)", "expansion", 1.5, 2.0,
            steps=[
                _step(1, "2x^2 - 10x + 3x - 15", "partial", 1.5,
                      "All four terms are correct but never combined — -10x and 3x should be simplified to -7x"),
            ],
            scheme=[_scheme(1, "2x^2 - 7x - 15", 2.0, "Expand using FOIL, then combine like terms")],
            feedback=[
                _fb(1, "All four expanded terms are individually correct (2x^2, -10x, +3x, -15).",
                    missing="The two middle terms -10x and +3x are like terms and must be combined: -10x+3x=-7x, giving 2x^2-7x-15.",
                    deduction="0.5 mark deducted because the expression was left unsimplified.",
                    improve="After expanding with FOIL, always finish by combining any like terms."),
            ],
            topic="expanding_brackets", difficulty="medium",
        ),
        _example(
            "Expand (x - 4)(x - 4)", "expansion", 0.0, 2.0,
            steps=[
                _step(1, "x^2 - 16", "incorrect", 0.0,
                      "(x-4)(x-4) is a perfect square, not a difference of squares — the two brackets are identical, not conjugates"),
            ],
            scheme=[_scheme(1, "x^2 - 8x + 16", 2.0, "Expand fully: x^2 - 4x - 4x + 16")],
            feedback=[
                _fb(1, "You correctly squared the first and last terms (x^2 and 16).",
                    missing="Difference of squares applies to (a-b)(a+b), NOT (a-b)(a-b). Here both brackets are identical, so this must be expanded as a normal product: x^2-4x-4x+16 = x^2-8x+16. The middle term -8x was omitted entirely.",
                    deduction="Full marks lost because the middle cross-terms were skipped — this is not a difference-of-squares pattern.",
                    improve="Only use the a^2-b^2 shortcut when the two brackets are conjugates (one + and one -). Otherwise, expand fully using FOIL."),
            ],
            topic="expanding_brackets", difficulty="medium",
        ),
    ]


def _difference_of_squares_topup_examples() -> List[Dict]:
    return [
        _example(
            "Factorise 4x^2 - 25", "difference of squares", 2.0, 2.0,
            steps=[_step(1, "(2x - 5)(2x + 5)", "correct", 2.0)],
            scheme=[_scheme(1, "(2x - 5)(2x + 5)", 2.0, "Recognise as a^2-b^2 with a=2x, b=5")],
            feedback=[_fb(1, "Correct — (2x)^2=4x^2 and 5^2=25.", improve="Full marks.")],
            topic="difference_of_squares", difficulty="easy",
        ),
        _example(
            "Factorise 5x^2 - 20", "common factor + difference of squares", 1.5, 2.0,
            steps=[
                _step(1, "5(x^2 - 4)", "correct", 1.0),
                _step(2, "5(x^2 - 4)", "partial", 0.5,
                      "Factorisation not complete — x^2-4 is itself a difference of squares"),
            ],
            scheme=[
                _scheme(1, "5(x^2 - 4)", 1.0, "Factor out the common factor 5"),
                _scheme(2, "5(x - 2)(x + 2)", 1.0, "Factorise the remaining difference of squares"),
            ],
            feedback=[
                _fb(1, "Correctly factored out the common factor 5."),
                _fb(2, "You correctly factored out the common factor 5.",
                    missing="x^2-4 is a difference of squares and factorises further into (x-2)(x+2).",
                    deduction="0.5 mark deducted because the factorisation was not taken to its fully factored form.",
                    improve="After factoring out a common term, always check whether what remains can be factorised further."),
            ],
            topic="difference_of_squares", difficulty="medium",
        ),
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Deepen still-thin topics, skewed toward partial/incorrect and
# difficult to correct the corpus-wide validity/difficulty balance.
# ─────────────────────────────────────────────────────────────────────────────

def _sum_diff_cubes_deepen_examples() -> List[Dict]:
    return [
        _example(
            "Factorise x^3 - 64", "difference of cubes", 2.0, 2.0,
            steps=[_step(1, "(x - 4)(x^2 + 4x + 16)", "correct", 2.0)],
            scheme=[_scheme(1, "(x - 4)(x^2 + 4x + 16)", 2.0, "Apply a^3-b^3=(a-b)(a^2+ab+b^2) with b=4")],
            feedback=[_fb(1, "Correct application of the difference-of-cubes identity.", improve="Full marks.")],
            topic="sum_difference_of_cubes", difficulty="medium",
        ),
        _example(
            "Factorise 8x^3 - 27", "difference of cubes", 0.0, 2.0,
            steps=[
                _step(1, "(2x - 3)(4x^2 + 3x + 9)", "incorrect", 0.0,
                      "Middle term error: ab = (2x)(3) = 6x, not 3x"),
            ],
            scheme=[_scheme(1, "(2x - 3)(4x^2 + 6x + 9)", 2.0, "Apply a^3-b^3=(a-b)(a^2+ab+b^2) with a=2x, b=3")],
            feedback=[
                _fb(1, "You correctly identified a=2x and b=3, and got a^2=4x^2 and b^2=9 right.",
                    missing="The middle term is ab = (2x)(3) = 6x, not 3x.",
                    deduction="Full marks lost because the middle term of the trinomial was computed incorrectly.",
                    improve="In a^3∓b^3=(a∓b)(a^2±ab+b^2), carefully compute ab as the full product of a and b, not just b."),
            ],
            topic="sum_difference_of_cubes", difficulty="difficult",
        ),
        _example(
            "Factorise x^3 - 1", "difference of cubes", 2.0, 2.0,
            steps=[
                _step(1, "a = x, b = 1", "correct", 1.0),
                _step(2, "(x - 1)(x^2 + x + 1)", "correct", 1.0),
            ],
            scheme=[
                _scheme(1, "a=x, b=1", 1.0, "Identify a and b for the identity a^3-b^3"),
                _scheme(2, "(x-1)(x^2+x+1)", 1.0, "Apply a^3-b^3=(a-b)(a^2+ab+b^2)"),
            ],
            feedback=[
                _fb(1, "Correctly identified a=x and b=1."),
                _fb(2, "Correct application of the difference-of-cubes identity.", improve="Full marks."),
            ],
            topic="sum_difference_of_cubes", difficulty="easy",
        ),
    ]


def _quadratic_formula_deepen_examples() -> List[Dict]:
    return [
        _example(
            "Solve 2x^2 - 4x - 6 = 0 using the quadratic formula", "quadratic formula", 2.0, 2.0,
            steps=[
                _step(1, "x = (4 ± √(16 + 48))/4 = (4 ± 8)/4", "correct", 1.0),
                _step(2, "x = 3 or x = -1", "correct", 1.0),
            ],
            scheme=[
                _scheme(1, "(4±8)/4", 1.0, "Apply the quadratic formula with a=2, b=-4, c=-6"),
                _scheme(2, "x=3 or x=-1", 1.0, "Evaluate both cases"),
            ],
            feedback=[
                _fb(1, "Correctly applied the quadratic formula and simplified the discriminant."),
                _fb(2, "Correctly evaluated both roots.", improve="Full marks."),
            ],
            topic="quadratic_formula", difficulty="medium",
        ),
        _example(
            "Solve x^2 + 2x - 4 = 0 using the quadratic formula", "quadratic formula", 1.5, 2.0,
            steps=[
                _step(1, "x = (-2 ± √(4 + 16))/2 = (-2 ± √20)/2", "correct", 1.0),
                _step(2, "x = (-2 ± √20)/2", "partial", 0.5,
                      "√20 not simplified — should be simplified to 2√5, giving x = -1 ± √5"),
            ],
            scheme=[
                _scheme(1, "(-2±√20)/2", 1.0, "Apply the quadratic formula with a=1, b=2, c=-4"),
                _scheme(2, "x = -1 ± √5", 1.0, "Simplify √20 = 2√5 and reduce the fraction"),
            ],
            feedback=[
                _fb(1, "Correctly applied the quadratic formula and computed the discriminant."),
                _fb(2, "Correct value under the square root.",
                    missing="√20 simplifies to 2√5 (since 20=4×5), giving x=(-2±2√5)/2 = -1±√5.",
                    deduction="0.5 mark deducted because the surd was left unsimplified.",
                    improve="Always simplify a surd to its simplest form before finalising an answer."),
            ],
            topic="quadratic_formula", difficulty="difficult",
        ),
        _example(
            "Solve 3x^2 + x - 2 = 0 using the quadratic formula", "quadratic formula", 0.0, 3.0,
            steps=[
                _step(1, "x = (1 ± √(1 + 24))/6 = (1 ± 5)/6", "incorrect", 0.0,
                      "Sign error: the formula uses -b; here b=1, so -b=-1, not +1"),
                _step(2, "x = 1 or x = -2/3", "incorrect", 0.0, "Follows from the sign error"),
            ],
            scheme=[
                _scheme(1, "(-1±5)/6", 1.5, "Apply the quadratic formula with a=3, b=1, c=-2"),
                _scheme(2, "x = 2/3 or x = -1", 1.5, "Evaluate both cases"),
            ],
            feedback=[
                _fb(1, "You correctly computed the discriminant (1+24=25) and its square root (5).",
                    missing="The quadratic formula numerator is -b ± √..., and here b=1, so -b=-1, not +1.",
                    deduction="Full marks lost because +b was used instead of -b."),
                _fb(2, "Your evaluation correctly followed from your Step 1 expression.",
                    missing="Because Step 1 had the sign error, these roots are wrong. Using -b=-1: x=(-1±5)/6, giving x=2/3 or x=-1.",
                    deduction="Marks lost because this follows from the earlier sign error.",
                    improve="Always double-check the sign of b before substituting into -b in the quadratic formula."),
            ],
            topic="quadratic_formula", difficulty="difficult",
        ),
    ]


def _completing_square_deepen_examples() -> List[Dict]:
    return [
        _example(
            "Solve x^2 - 2x - 8 = 0 by completing the square", "completing the square", 3.0, 3.0,
            steps=[
                _step(1, "(x - 1)^2 - 9 = 0", "correct", 1.0),
                _step(2, "(x - 1)^2 = 9", "correct", 1.0),
                _step(3, "x = 4 or x = -2", "correct", 1.0),
            ],
            scheme=[
                _scheme(1, "(x-1)^2 - 9 = 0", 1.0, "Complete the square (half of -2 is -1)"),
                _scheme(2, "(x-1)^2 = 9", 1.0, "Simplify constants"),
                _scheme(3, "x = 4 or x = -2", 1.0, "Take the square root and solve"),
            ],
            feedback=[
                _fb(1, "Correctly completed the square using half of the x-coefficient."),
                _fb(2, "Correctly simplified the constants."),
                _fb(3, "Both roots correctly stated.", improve="Full marks."),
            ],
            topic="completing_the_square", difficulty="medium",
        ),
        _example(
            "Solve x^2 + 8x + 10 = 0 by completing the square", "completing the square", 3.0, 4.0,
            steps=[
                _step(1, "(x + 4)^2 - 6 = 0", "correct", 1.0),
                _step(2, "(x + 4)^2 = 6", "correct", 1.0),
                _step(3, "x + 4 = √6 → x = -4 + √6", "partial", 1.0,
                      "Missing the negative case — should be x = -4 ± √6"),
            ],
            scheme=[
                _scheme(1, "(x+4)^2 - 6 = 0", 1.0, "Complete the square (half of 8 is 4)"),
                _scheme(2, "(x+4)^2 = 6", 1.0, "Simplify constants"),
                _scheme(3, "x = -4 ± √6", 2.0, "Take BOTH square roots"),
            ],
            feedback=[
                _fb(1, "Correctly completed the square using half of the x-coefficient."),
                _fb(2, "Correctly simplified the constants."),
                _fb(3, "x = -4 + √6 is one correct solution.",
                    missing="Taking a square root always gives two cases: x+4=±√6. The missing case is x+4=-√6, giving x=-4-√6.",
                    deduction="1 mark deducted for the missing negative root.",
                    improve="Whenever you reach (x-a)^2=k, always write x-a=±√k to capture both solutions."),
            ],
            topic="completing_the_square", difficulty="difficult",
        ),
        _example(
            "Solve 2x^2 - 8x + 3 = 0 by completing the square", "completing the square", 0.0, 4.0,
            steps=[
                _step(1, "x^2 - 4x + 3 = 0", "incorrect", 0.0,
                      "Dividing 2x^2-8x+3 by 2 gives x^2-4x+1.5, not x^2-4x+3 — the constant term must also be divided by 2"),
            ],
            scheme=[_scheme(1, "x^2 - 4x + 1.5 = 0", 4.0, "Divide EVERY term, including the constant, by the leading coefficient 2")],
            feedback=[
                _fb(1, "You correctly divided the x^2 and x terms by 2.",
                    missing="The constant term 3 must ALSO be divided by 2, giving 1.5, not left as 3. The correct starting equation is x^2-4x+1.5=0.",
                    deduction="Full marks lost because the constant term was not divided along with the other terms.",
                    improve="When the leading coefficient isn't 1, divide EVERY term in the equation by it before completing the square."),
            ],
            topic="completing_the_square", difficulty="difficult",
        ),
    ]


def _surds_deepen_examples() -> List[Dict]:
    return [
        _example(
            "Simplify √72", "surds", 1.0, 1.0,
            steps=[_step(1, "√(36 × 2) = 6√2", "correct", 1.0)],
            scheme=[_scheme(1, "6√2", 1.0, "Find the largest perfect-square factor of 72")],
            feedback=[_fb(1, "Correctly identified 36 as the largest perfect-square factor.", improve="Full marks.")],
            topic="surds", difficulty="easy",
        ),
        _example(
            "Simplify 3√8 - √2", "surds", 0.0, 2.0,
            steps=[
                _step(1, "3√8 - √2 = 2√6", "incorrect", 0.0,
                      "Cannot subtract radicands directly — √8 must be simplified first, then combined with √2 as like surds"),
            ],
            scheme=[_scheme(1, "5√2", 2.0, "Simplify √8=2√2 first, then combine like surds: 3(2√2)-√2=6√2-√2")],
            feedback=[
                _fb(1, "You attempted to combine the two surds into one expression.",
                    missing="You cannot subtract the numbers inside different-looking square roots directly. First simplify √8=2√2, so 3√8=6√2. THEN combine like surds: 6√2-√2=5√2.",
                    deduction="Full marks lost because unlike-looking surds were combined without first simplifying to a common surd.",
                    improve="Before adding or subtracting surds, always simplify each one to its simplest form so you can identify genuinely like surds."),
            ],
            topic="surds", difficulty="difficult",
        ),
        _example(
            "Simplify (√5)^2 + 3", "surds", 1.0, 1.0,
            steps=[_step(1, "5 + 3 = 8", "correct", 1.0)],
            scheme=[_scheme(1, "8", 1.0, "(√5)^2 = 5, then add 3")],
            feedback=[_fb(1, "Correctly recognised that squaring a square root cancels it.", improve="Full marks.")],
            topic="surds", difficulty="easy",
        ),
    ]


def _linear_inequalities_deepen_examples() -> List[Dict]:
    return [
        _example(
            "Solve 3x - 4 ≤ 11", "linear inequality", 1.0, 1.0,
            steps=[_step(1, "3x ≤ 15 → x ≤ 5", "correct", 1.0)],
            scheme=[_scheme(1, "x ≤ 5", 1.0, "Add 4, then divide by 3")],
            feedback=[_fb(1, "Correctly isolated x — no sign flip needed since dividing by a positive.", improve="Full marks.")],
            topic="linear_inequalities", difficulty="easy",
        ),
        _example(
            "Solve 4 - x ≥ -2", "linear inequality", 1.0, 2.0,
            steps=[
                _step(1, "4 - x ≥ -2 → -x ≥ -6", "correct", 1.0),
                _step(2, "x ≥ 6", "incorrect", 0.0,
                      "Dividing/multiplying by -1 flips the inequality: -x≥-6 means x≤6, not x≥6"),
            ],
            scheme=[
                _scheme(1, "-x ≥ -6", 1.0, "Subtract 4 from both sides"),
                _scheme(2, "x ≤ 6", 1.0, "Multiply by -1 and flip the inequality"),
            ],
            feedback=[
                _fb(1, "Correctly subtracted 4 from both sides."),
                _fb(2, "You correctly isolated -x.",
                    missing="Multiplying or dividing both sides by -1 flips the inequality sign: -x≥-6 becomes x≤6, not x≥6.",
                    deduction="Full marks lost because the inequality sign was not flipped.",
                    improve="Always flip the inequality sign whenever you multiply or divide both sides by a negative number."),
            ],
            topic="linear_inequalities", difficulty="medium",
        ),
        _example(
            "Solve -2(x - 3) < 10", "linear inequality", 1.0, 2.0,
            steps=[
                _step(1, "-2x + 6 < 10 → -2x < 4", "correct", 1.0),
                _step(2, "x < -2", "incorrect", 0.0,
                      "Dividing by -2 flips the inequality: should be x > -2, not x < -2"),
            ],
            scheme=[
                _scheme(1, "-2x < 4", 1.0, "Distribute and isolate the x term"),
                _scheme(2, "x > -2", 1.0, "Divide by -2 and flip the inequality"),
            ],
            feedback=[
                _fb(1, "Correctly distributed the -2 and isolated the x term."),
                _fb(2, "You correctly divided both sides by -2.",
                    missing="Dividing by a negative number flips the inequality: -2x<4 means x>-2, not x<-2.",
                    deduction="Full marks lost because the inequality sign was not flipped.",
                    improve="Always flip the inequality sign whenever dividing or multiplying by a negative number."),
            ],
            topic="linear_inequalities", difficulty="medium",
        ),
    ]


def _quadratic_inequalities_deepen_examples() -> List[Dict]:
    return [
        _example(
            "Solve x^2 - x - 6 ≤ 0", "quadratic inequality", 2.0, 2.0,
            steps=[
                _step(1, "(x - 3)(x + 2) ≤ 0", "correct", 1.0),
                _step(2, "-2 ≤ x ≤ 3", "correct", 1.0),
            ],
            scheme=[
                _scheme(1, "(x-3)(x+2) ≤ 0", 1.0, "Factorise the quadratic"),
                _scheme(2, "-2 ≤ x ≤ 3", 1.0, "Identify the region between the roots"),
            ],
            feedback=[
                _fb(1, "Correct factorisation."),
                _fb(2, "Correctly identified the region where the product is negative or zero.", improve="Full marks."),
            ],
            topic="quadratic_inequalities", difficulty="medium",
        ),
        _example(
            "Solve x^2 + 3x > 0", "quadratic inequality", 1.5, 2.0,
            steps=[
                _step(1, "x(x + 3) > 0", "correct", 1.0),
                _step(2, "x > 0", "partial", 0.5,
                      "Missing the other region — since this is '>' (outside the roots), x < -3 is also valid"),
            ],
            scheme=[
                _scheme(1, "x(x+3) > 0", 1.0, "Factorise the quadratic"),
                _scheme(2, "x < -3 or x > 0", 1.0, "Identify BOTH regions outside the roots"),
            ],
            feedback=[
                _fb(1, "Correct factorisation."),
                _fb(2, "x > 0 is one correct region.",
                    missing="Since the inequality is '> 0' (outside the roots), the region x < -3 is also part of the solution.",
                    deduction="0.5 mark deducted for the missing second region.",
                    improve="For a quadratic inequality with two real roots, sketch a sign diagram or the parabola to find ALL regions satisfying the inequality."),
            ],
            topic="quadratic_inequalities", difficulty="difficult",
        ),
        _example(
            "Solve x^2 ≤ 16", "quadratic inequality", 0.0, 2.0,
            steps=[
                _step(1, "x ≤ 4", "incorrect", 0.0,
                      "x^2≤16 means -4≤x≤4 — taking a square root of an inequality requires considering both bounds"),
            ],
            scheme=[_scheme(1, "-4 ≤ x ≤ 4", 2.0, "Take the square root of both sides, keeping both bounds")],
            feedback=[
                _fb(1, "You correctly found the upper bound x=4.",
                    missing="x^2≤16 means -4≤x≤4, not just x≤4 — taking the square root of an inequality (with a positive right-hand side) always gives a range between the negative and positive root.",
                    deduction="Full marks lost because the lower bound -4≤x was omitted.",
                    improve="For x^2≤k (k>0), the solution is always -√k≤x≤√k — never just the upper bound."),
            ],
            topic="quadratic_inequalities", difficulty="medium",
        ),
    ]


def _simultaneous_elimination_deepen_examples() -> List[Dict]:
    return [
        _example(
            "Solve: 3x + 2y = 16, x - 2y = 0", "elimination", 2.0, 2.0,
            steps=[
                _step(1, "Adding: 4x = 16 → x = 4", "correct", 1.0),
                _step(2, "x = 2y → y = 2", "correct", 1.0),
            ],
            scheme=[
                _scheme(1, "x = 4", 1.0, "Add the equations to eliminate y"),
                _scheme(2, "y = 2", 1.0, "Substitute back to find y"),
            ],
            feedback=[
                _fb(1, "Adding correctly eliminates y since +2y and -2y cancel."),
                _fb(2, "Correct substitution to find y.", improve="Full marks. Verify: 3(4)+2(2)=16 ✓."),
            ],
            topic="simultaneous_elimination", difficulty="easy",
        ),
        _example(
            "Solve: 4x - y = 5, 2x + y = 7", "elimination", 1.0, 2.0,
            steps=[
                _step(1, "Adding: 6x = 12 → x = 2", "correct", 1.0),
                _step(2, "y = 7 - 2(2) = 4", "incorrect", 0.0, "Arithmetic slip: 7-4=3, not 4"),
            ],
            scheme=[
                _scheme(1, "x = 2", 1.0, "Add the equations to eliminate y"),
                _scheme(2, "y = 3", 1.0, "Substitute back to find y"),
            ],
            feedback=[
                _fb(1, "Adding correctly eliminates y."),
                _fb(2, "Correct method of substituting back.",
                    missing="7 - 2(2) = 7 - 4 = 3, not 4.",
                    deduction="Marks lost for the arithmetic slip despite the correct method.",
                    improve="Double-check basic subtraction. Verify: 4(2)-3=5 ✓."),
            ],
            topic="simultaneous_elimination", difficulty="easy",
        ),
        _example(
            "Solve: 5x + 3y = 19, 2x - 3y = 2", "elimination", 2.0, 2.0,
            steps=[
                _step(1, "Adding: 7x = 21 → x = 3", "correct", 1.0),
                _step(2, "15 + 3y = 19 → y = 4/3", "correct", 1.0),
            ],
            scheme=[
                _scheme(1, "x = 3", 1.0, "Add the equations to eliminate y"),
                _scheme(2, "y = 4/3", 1.0, "Substitute back to find y"),
            ],
            feedback=[
                _fb(1, "Adding correctly eliminates y since +3y and -3y cancel."),
                _fb(2, "Correctly substituted back and solved for y.", improve="Full marks — a fractional but correct answer."),
            ],
            topic="simultaneous_elimination", difficulty="difficult",
        ),
    ]


def _function_composition_deepen_examples() -> List[Dict]:
    return [
        _example(
            "If f(x) = x + 5 and g(x) = x^2, find (f∘g)(-2)", "function composition", 2.0, 2.0,
            steps=[
                _step(1, "g(-2) = 4", "correct", 1.0),
                _step(2, "f(4) = 9", "correct", 1.0),
            ],
            scheme=[
                _scheme(1, "g(-2) = 4", 1.0, "Evaluate the inner function g first"),
                _scheme(2, "f(4) = 9", 1.0, "Apply f to the result"),
            ],
            feedback=[
                _fb(1, "Correctly evaluated g(-2), noting (-2)^2=4 (positive)."),
                _fb(2, "Correctly applied f to the result.", improve="Full marks."),
            ],
            topic="function_composition", difficulty="medium",
        ),
        _example(
            "If f(x) = 2x, g(x) = x + 1, find (g∘g)(3)", "function composition", 1.0, 2.0,
            steps=[
                _step(1, "g(3) = 3 + 1 = 4", "correct", 1.0),
                _step(2, "g(4) = 4 + 1 = 6", "incorrect", 0.0, "Arithmetic slip: 4+1=5, not 6"),
            ],
            scheme=[
                _scheme(1, "g(3) = 4", 1.0, "Evaluate the inner g(3)"),
                _scheme(2, "g(4) = 5", 1.0, "Apply g again to the result"),
            ],
            feedback=[
                _fb(1, "Correctly evaluated the inner g(3)."),
                _fb(2, "Correct method of applying g a second time.",
                    missing="4 + 1 = 5, not 6.",
                    deduction="Marks lost for the arithmetic slip despite the correct method.",
                    improve="Double-check basic addition when applying a function repeatedly."),
            ],
            topic="function_composition", difficulty="easy",
        ),
    ]


def _inverse_functions_deepen_examples() -> List[Dict]:
    return [
        _example(
            "Find the inverse of f(x) = 4x - 8", "inverse function", 2.0, 2.0,
            steps=[
                _step(1, "x = 4y - 8 → 4y = x + 8", "correct", 1.0),
                _step(2, "y = (x + 8)/4", "correct", 1.0),
            ],
            scheme=[
                _scheme(1, "4y = x + 8", 1.0, "Swap x and y, then rearrange"),
                _scheme(2, "y = (x+8)/4", 1.0, "Solve for y"),
            ],
            feedback=[
                _fb(1, "Correctly swapped x and y and rearranged."),
                _fb(2, "Correctly solved for y.", improve="Full marks."),
            ],
            topic="inverse_functions", difficulty="medium",
        ),
        _example(
            "Find the inverse of f(x) = x/2 + 3", "inverse function", 1.0, 2.0,
            steps=[
                _step(1, "x - 3 = y/2", "correct", 1.0),
                _step(2, "y = x - 3", "incorrect", 0.0, "Forgot to multiply both sides by 2 — should be y=2(x-3)=2x-6"),
            ],
            scheme=[
                _scheme(1, "x - 3 = y/2", 1.0, "Swap x and y, then isolate y/2"),
                _scheme(2, "y = 2x - 6", 1.0, "Multiply both sides by 2"),
            ],
            feedback=[
                _fb(1, "Correctly swapped x and y and isolated y/2."),
                _fb(2, "You correctly reached x-3=y/2.",
                    missing="To finish isolating y, multiply both sides by 2: y=2(x-3)=2x-6, not just x-3.",
                    deduction="Full marks lost because the final multiplication step was skipped.",
                    improve="Always complete every remaining operation needed to fully isolate y."),
            ],
            topic="inverse_functions", difficulty="medium",
        ),
    ]


def _gradient_intercept_deepen_examples() -> List[Dict]:
    return [
        _example(
            "A line has y-intercept -3 and passes through (2, 5). Find its equation.",
            "gradient and intercept", 2.0, 2.0,
            steps=[
                _step(1, "5 = 2m - 3 → 2m = 8", "correct", 1.0),
                _step(2, "m = 4, y = 4x - 3", "correct", 1.0),
            ],
            scheme=[
                _scheme(1, "2m = 8", 1.0, "Substitute the point and intercept into y=mx+c"),
                _scheme(2, "y = 4x - 3", 1.0, "Solve for m"),
            ],
            feedback=[
                _fb(1, "Correctly substituted the known point and intercept."),
                _fb(2, "Correctly solved for the gradient.", improve="Full marks. Verify: 4(2)-3=5 ✓."),
            ],
            topic="gradient_intercept", difficulty="medium",
        ),
        _example(
            "Two lines y = 3x + 2 and y = 3x - 5 are drawn. State their relationship.",
            "gradient and intercept (parallel/perpendicular)", 0.0, 1.0,
            steps=[
                _step(1, "The lines are perpendicular", "incorrect", 0.0,
                      "Perpendicular lines have gradients that are negative reciprocals — these two lines share the SAME gradient, making them parallel"),
            ],
            scheme=[_scheme(1, "Parallel", 1.0, "Compare the gradients: both are 3")],
            feedback=[
                _fb(1, "You correctly compared the two equations.",
                    missing="Perpendicular lines have gradients that multiply to -1 (negative reciprocals). These two lines both have gradient 3 — equal gradients mean the lines are PARALLEL, not perpendicular.",
                    deduction="Full marks lost for stating the wrong relationship.",
                    improve="Same gradient → parallel. Gradients that are negative reciprocals (product = -1) → perpendicular."),
            ],
            topic="gradient_intercept", difficulty="easy",
        ),
    ]


def _sequences_nth_term_deepen_examples() -> List[Dict]:
    return [
        _example(
            "Find the nth term of the sequence 20, 17, 14, 11, ...", "sequences (nth term, decreasing)", 1.0, 1.0,
            steps=[_step(1, "d = -3, so nth term = 23 - 3n", "correct", 1.0)],
            scheme=[_scheme(1, "23 - 3n", 1.0, "Find the (negative) common difference and apply a+(n-1)d, simplified")],
            feedback=[_fb(1, "Correctly identified the negative common difference and formed the nth term expression.",
                           improve="Full marks. Check: n=1 gives 20 ✓.")],
            topic="sequences_nth_term", difficulty="medium",
        ),
        _example(
            "Find the 15th term of the sequence 3, 7, 11, 15, ...", "sequences (nth term)", 0.0, 2.0,
            steps=[
                _step(1, "3 + 15(4) = 63", "incorrect", 0.0,
                      "nth term formula is a+(n-1)d, not a+nd — should be 3+14(4)"),
            ],
            scheme=[_scheme(1, "59", 2.0, "Apply a+(n-1)d with a=3, d=4, n=15")],
            feedback=[
                _fb(1, "You correctly identified the common difference d=4.",
                    missing="The nth term formula is a+(n-1)d, not a+nd. For the 15th term, use (n-1)=14: 3+14(4)=59, not 3+15(4)=63.",
                    deduction="Full marks lost because the '-1' in the nth term formula was omitted.",
                    improve="Always double-check: the nth term formula is a + (n-1)d — the exponent on d is (n-1), not n."),
            ],
            topic="sequences_nth_term", difficulty="medium",
        ),
    ]


def _ratio_proportion_deepen_examples() -> List[Dict]:
    return [
        _example(
            "Divide 45 in the ratio 4:5", "ratio and proportion (three-step)", 2.0, 2.0,
            steps=[_step(1, "total parts = 9, each part = 45/9 = 5, so 20 : 25", "correct", 2.0)],
            scheme=[_scheme(1, "20 : 25", 2.0, "Find the value of one part, then scale up both parts")],
            feedback=[_fb(1, "Correctly found the value of one part and scaled up both parts.",
                           improve="Full marks. Verify: 20+25=45 ✓.")],
            topic="ratio_proportion", difficulty="easy",
        ),
        _example(
            "y is inversely proportional to x. When x=3, y=8. Find y when x=4.",
            "ratio and proportion (inverse variation)", 0.0, 2.0,
            steps=[
                _step(1, "y = kx → k = 8/3", "incorrect", 0.0,
                      "This is INVERSE proportion (y=k/x), not direct (y=kx) — 'inversely proportional' means as x increases, y decreases"),
            ],
            scheme=[_scheme(1, "y = 6", 2.0, "y=k/x, so k=xy=3×8=24; when x=4, y=24/4=6")],
            feedback=[
                _fb(1, "You correctly set up an equation involving a constant k.",
                    missing="'Inversely proportional' means y=k/x, not y=kx. Here k=xy=3×8=24, so y=24/x. When x=4, y=24/4=6.",
                    deduction="Full marks lost because direct proportion was used instead of inverse proportion.",
                    improve="Direct proportion: y=kx (y increases with x). Inverse proportion: y=k/x (y decreases as x increases). Always check which the question describes."),
            ],
            topic="ratio_proportion", difficulty="difficult",
        ),
    ]


def _absolute_value_deepen_examples() -> List[Dict]:
    return [
        _example(
            "Solve |3x| = 12", "absolute value equation", 2.0, 2.0,
            steps=[_step(1, "3x = 12 or 3x = -12 → x = 4 or x = -4", "correct", 2.0)],
            scheme=[_scheme(1, "x = 4 or x = -4", 2.0, "Split into the positive and negative case")],
            feedback=[_fb(1, "Correctly split into both cases and solved.", improve="Full marks.")],
            topic="absolute_value_equations", difficulty="easy",
        ),
        _example(
            "Solve |x - 1| = |2x + 3|", "absolute value equation (both sides)", 2.0, 2.0,
            steps=[
                _step(1, "Case 1: x - 1 = 2x + 3 → x = -4", "correct", 1.0),
                _step(2, "Case 2: x - 1 = -(2x + 3) → x = -2/3", "correct", 1.0),
            ],
            scheme=[
                _scheme(1, "x = -4", 1.0, "Case where both sides have the same sign"),
                _scheme(2, "x = -2/3", 1.0, "Case where the sides have opposite signs"),
            ],
            feedback=[
                _fb(1, "Correctly solved the case where both expressions are directly equal."),
                _fb(2, "Correctly solved the case where one expression is the negative of the other.",
                    improve="Full marks — both required cases were considered."),
            ],
            topic="absolute_value_equations", difficulty="difficult",
        ),
    ]


def _exponent_equations_deepen_examples() -> List[Dict]:
    return [
        _example(
            "Solve 5^(2x) = 125", "exponent equation", 2.0, 2.0,
            steps=[
                _step(1, "125 = 5^3", "correct", 1.0),
                _step(2, "2x = 3 → x = 1.5", "correct", 1.0),
            ],
            scheme=[
                _scheme(1, "125 = 5^3", 1.0, "Rewrite 125 as a power of 5"),
                _scheme(2, "x = 1.5", 1.0, "Equate exponents and solve"),
            ],
            feedback=[
                _fb(1, "Correctly rewrote 125 as a power of 5."),
                _fb(2, "Correctly equated exponents and solved.", improve="Full marks."),
            ],
            topic="exponent_equations", difficulty="medium",
        ),
        _example(
            "Solve 2^x × 2^3 = 2^10", "exponent equation", 0.0, 2.0,
            steps=[
                _step(1, "2^(3x) = 2^10 → 3x = 10 → x = 10/3", "incorrect", 0.0,
                      "When multiplying powers of the same base, exponents ADD, not multiply: 2^x×2^3=2^(x+3)"),
            ],
            scheme=[_scheme(1, "x = 7", 2.0, "Combine using 2^x×2^3=2^(x+3), so x+3=10")],
            feedback=[
                _fb(1, "You correctly recognised that the two powers on the left must be combined into one.",
                    missing="When multiplying powers of the same base, exponents ADD: 2^x×2^3=2^(x+3), not 2^(3x). So x+3=10, giving x=7.",
                    deduction="Full marks lost because the index law for multiplication was applied incorrectly.",
                    improve="Remember a^m × a^n = a^(m+n) — always ADD exponents when multiplying powers of the same base, never multiply them."),
            ],
            topic="exponent_equations", difficulty="medium",
        ),
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Heavily skewed toward partial validity to correct the
# corpus-wide correct/partial/incorrect balance.
# ─────────────────────────────────────────────────────────────────────────────

def _validity_rebalance_examples() -> List[Dict]:
    return [
        _example(
            "Add (2x^2 - 3x + 1) and (x^2 + 3x - 4)", "polynomial addition", 2.5, 3.0,
            steps=[
                _step(1, "2x^2 + x^2 = 3x^2", "correct", 1.0),
                _step(2, "-3x + 3x = 0", "correct", 1.0),
                _step(3, "1 - 4", "partial", 0.5, "Not evaluated to a final number — should be -3"),
            ],
            scheme=[
                _scheme(1, "3x^2", 1.0, "Combine the x^2 terms"),
                _scheme(2, "0", 1.0, "Combine the x terms (they cancel)"),
                _scheme(3, "-3", 1.0, "Combine the constant terms"),
            ],
            feedback=[
                _fb(1, "Correctly combined the x^2 terms."),
                _fb(2, "Correctly noted the x terms cancel to 0."),
                _fb(3, "Correct method of combining the constants.",
                    missing="1 - 4 must be evaluated to a final number: -3.",
                    deduction="0.5 mark deducted because the arithmetic was left unevaluated.",
                    improve="Always finish arithmetic to a final simplified number. Final answer: 3x^2 - 3."),
            ],
            topic="polynomial_addition_subtraction", difficulty="medium",
        ),
        _example(
            "Subtract (x^3 - 2x + 5) from (2x^3 + x^2 - 3)", "polynomial subtraction", 2.0, 3.0,
            steps=[
                _step(1, "2x^3 + x^2 - 3 - x^3 + 2x - 5", "correct", 1.5),
                _step(2, "x^3 + 2x - 8", "partial", 0.5,
                      "Dropped the x^2 term — it has no like term but must still appear in the final answer"),
            ],
            scheme=[
                _scheme(1, "2x^3+x^2-3-x^3+2x-5", 1.5, "Distribute the negative sign to every term"),
                _scheme(2, "x^3 + x^2 + 2x - 8", 1.5, "Combine like terms, keeping unmatched terms"),
            ],
            feedback=[
                _fb(1, "Correctly distributed the negative sign to every term of the subtracted polynomial."),
                _fb(2, "Correctly combined the x^3 and constant terms.",
                    missing="The x^2 term has no like term to combine with, but it must still appear in the final answer: x^3+x^2+2x-8, not x^3+2x-8.",
                    deduction="0.5 mark deducted because the unmatched x^2 term was dropped.",
                    improve="Terms with no matching like term still carry through to the final answer unchanged — don't drop them."),
            ],
            topic="polynomial_addition_subtraction", difficulty="medium",
        ),
        _example(
            "Multiply x(x + 3)(x - 2)", "polynomial multiplication", 1.5, 2.0,
            steps=[
                _step(1, "(x + 3)(x - 2) = x^2 + x - 6", "correct", 1.0),
                _step(2, "x^2 + x - 6", "partial", 0.5,
                      "Forgot to multiply by the outer x — final answer should be x^3 + x^2 - 6x"),
            ],
            scheme=[
                _scheme(1, "x^2 + x - 6", 1.0, "Expand the two brackets first"),
                _scheme(2, "x^3 + x^2 - 6x", 1.0, "Multiply the result by the outer x"),
            ],
            feedback=[
                _fb(1, "Correctly expanded the two brackets."),
                _fb(2, "You correctly expanded (x+3)(x-2).",
                    missing="The outer x still needs to be multiplied across this result: x(x^2+x-6) = x^3+x^2-6x.",
                    deduction="Full marks lost for the final step because the outer x was never applied.",
                    improve="With three factors, expand two of them first, then multiply the ENTIRE result by the remaining factor — don't stop early."),
            ],
            topic="polynomial_multiplication", difficulty="difficult",
        ),
        _example(
            "Multiply (x + 2)^2", "polynomial multiplication (squaring a bracket)", 1.5, 2.0,
            steps=[
                _step(1, "x^2 + 2x + 2x + 2^2", "correct", 1.0),
                _step(2, "x^2 + 4x + 2^2", "partial", 0.5, "2^2 not evaluated — should be x^2 + 4x + 4"),
            ],
            scheme=[
                _scheme(1, "x^2+2x+2x+2^2", 1.0, "Expand using (a+b)^2=a^2+2ab+b^2"),
                _scheme(2, "x^2 + 4x + 4", 1.0, "Evaluate 2^2 and combine like terms"),
            ],
            feedback=[
                _fb(1, "Correctly applied the (a+b)^2 expansion pattern."),
                _fb(2, "Correctly combined the middle terms (2x+2x=4x).",
                    missing="2^2 must be evaluated to a final number: 4.",
                    deduction="0.5 mark deducted because the constant term was left unevaluated.",
                    improve="Always finish arithmetic to a final simplified number."),
            ],
            topic="polynomial_multiplication", difficulty="easy",
        ),
        _example(
            "Divide (3x^2 + 12x) by 3x", "polynomial division", 1.5, 2.0,
            steps=[
                _step(1, "3x^2/3x = x", "correct", 1.0),
                _step(2, "12x/3 = 4x", "partial", 0.5,
                      "Only divided by 3, not by the full 3x — 12x/3x = 4"),
            ],
            scheme=[
                _scheme(1, "x", 1.0, "Divide the first term by 3x"),
                _scheme(2, "4", 1.0, "Divide the second term by 3x"),
            ],
            feedback=[
                _fb(1, "Correctly divided the first term by 3x."),
                _fb(2, "You correctly began dividing the second term.",
                    missing="The divisor is 3x, not just 3 — 12x ÷ 3x = 4 (the x cancels), not 12x ÷ 3 = 4x.",
                    deduction="0.5 mark deducted because only part of the divisor was applied.",
                    improve="When dividing by a term like 3x, divide by the WHOLE term, coefficient and variable together."),
            ],
            topic="polynomial_division", difficulty="medium",
        ),
        _example(
            "Divide (4x^2 - 9) by (2x - 3)", "polynomial division (by factorisation)", 0.0, 2.0,
            steps=[
                _step(1, "(2x - 3)(2x - 3)", "incorrect", 0.0,
                      "4x^2-9 is a difference of squares (2x-3)(2x+3), not a repeated factor (2x-3)(2x-3)"),
            ],
            scheme=[_scheme(1, "2x + 3", 2.0, "Factorise as a difference of squares, then cancel (2x-3)")],
            feedback=[
                _fb(1, "You correctly identified (2x-3) as one factor of the dividend.",
                    missing="4x^2-9 is a difference of squares: (2x-3)(2x+3), not (2x-3)(2x-3) — which would expand to 4x^2-12x+9, not 4x^2-9. The quotient is 2x+3.",
                    deduction="Full marks lost because the wrong factorisation pattern was used.",
                    improve="For a^2-b^2, always factorise as (a-b)(a+b), never as (a-b)(a-b)."),
            ],
            topic="polynomial_division", difficulty="medium",
        ),
        _example(
            "Simplify 10 - 3(x - 2)", "simplifying expressions", 1.5, 2.0,
            steps=[
                _step(1, "10 - 3x + 6", "correct", 1.0),
                _step(2, "10 - 3x + 6", "partial", 0.5,
                      "Constants not combined — should be simplified to 16 - 3x"),
            ],
            scheme=[
                _scheme(1, "10 - 3x + 6", 1.0, "Distribute the -3 across the bracket"),
                _scheme(2, "16 - 3x", 1.0, "Combine the constant terms"),
            ],
            feedback=[
                _fb(1, "Correctly distributed the -3 across the bracket."),
                _fb(2, "Correct distribution.",
                    missing="The constants 10 and 6 are like terms and must be combined: 10+6=16, giving 16-3x.",
                    deduction="0.5 mark deducted because the expression was left unsimplified.",
                    improve="After distributing, always finish by combining any remaining like terms."),
            ],
            topic="simplifying_expressions", difficulty="easy",
        ),
        _example(
            "Factorise 8x^3 + 12x^2", "common factor", 1.0, 2.0,
            steps=[
                _step(1, "4x(2x^2 + 3x)", "partial", 1.0,
                      "The full common factor is 4x^2, not just 4x — x^2 is common to both terms"),
            ],
            scheme=[_scheme(1, "4x^2(2x + 3)", 2.0, "Factor out the highest common factor, 4x^2")],
            feedback=[
                _fb(1, "You correctly identified 4 as the numeric common factor and pulled out one x.",
                    missing="Both terms (8x^3 and 12x^2) share x^2 as a common factor, not just x. The full factorisation is 4x^2(2x+3), not 4x(2x^2+3x).",
                    deduction="1 mark deducted because the factorisation is incomplete — an extra x was left inside the bracket unnecessarily.",
                    improve="Always check the HIGHEST power of a variable common to every term, not just one factor of it. Verify: 4x^2(2x+3)=8x^3+12x^2 ✓."),
            ],
            topic="factorising_common_factor", difficulty="medium",
        ),
        _example(
            "Factorise 3x^2 + 11x + 6", "factorisation by grouping", 2.0, 3.0,
            steps=[
                _step(1, "3x^2 + 9x + 2x + 6", "correct", 1.0),
                _step(2, "3x(x + 3) + 2(x + 3)", "correct", 1.0),
                _step(3, "(3x - 2)(x - 3)", "incorrect", 0.0,
                      "Sign error writing the final factors — should be (3x + 2)(x + 3), matching the positive grouped terms"),
            ],
            scheme=[
                _scheme(1, "3x^2+9x+2x+6", 1.0, "Split the middle term using factors of 3x6=18 that sum to 11: 9 and 2"),
                _scheme(2, "3x(x+3)+2(x+3)", 1.0, "Factor by grouping"),
                _scheme(3, "(3x+2)(x+3)", 1.0, "Factor out the common bracket (x+3)"),
            ],
            feedback=[
                _fb(1, "Correctly split the middle term using 9 and 2, factors of 18 that sum to 11."),
                _fb(2, "Correctly factored each pair of terms, revealing the common bracket (x+3)."),
                _fb(3, "You correctly factored out (x+3) from both grouped terms in Step 2.",
                    missing="Both grouped terms were positive (3x(x+3) and 2(x+3)), so the factors must keep those signs: (3x+2)(x+3), not (3x-2)(x-3).",
                    deduction="Full marks lost because the signs were flipped when writing the final factorisation.",
                    improve="Write the final factorisation directly from your grouping step — the signs inside each bracket must match what you already grouped, not be changed."),
            ],
            topic="factorising_quadratic", difficulty="difficult",
        ),
        _example(
            "Solve 5x + 2 = 3x + 10", "linear equation", 1.5, 2.0,
            steps=[
                _step(1, "2x = 8", "correct", 1.0),
                _step(2, "x = 8/2", "partial", 0.5, "Not simplified — should be x = 4"),
            ],
            scheme=[
                _scheme(1, "2x = 8", 1.0, "Collect x terms on one side, constants on the other"),
                _scheme(2, "x = 4", 1.0, "Divide both sides by 2 and simplify"),
            ],
            feedback=[
                _fb(1, "Correctly collected the x terms and constants."),
                _fb(2, "Correct method of dividing by 2.",
                    missing="8/2 must be evaluated to a final number: x=4.",
                    deduction="0.5 mark deducted because the fraction was left unsimplified.",
                    improve="Always finish arithmetic to a final simplified number."),
            ],
            topic="linear_equations", difficulty="easy",
        ),
        _example(
            "Solve: y = x - 2, y = x^2 - 4x + 2", "substitution (linear-quadratic)", 2.5, 3.0,
            steps=[
                _step(1, "x - 2 = x^2 - 4x + 2 → x^2 - 5x + 4 = 0", "correct", 1.0),
                _step(2, "(x - 1)(x - 4) = 0 → x = 1 or x = 4", "correct", 1.0),
                _step(3, "y = 1 - 2 = -1", "partial", 0.5,
                      "Only one solution pair found — the second point (4, 2) is missing"),
            ],
            scheme=[
                _scheme(1, "x^2-5x+4=0", 1.0, "Substitute to eliminate y"),
                _scheme(2, "x=1 or x=4", 1.0, "Factorise and solve"),
                _scheme(3, "(1,-1) and (4,2)", 1.0, "State BOTH solution pairs"),
            ],
            feedback=[
                _fb(1, "Correct substitution and rearrangement."),
                _fb(2, "Correct factorisation and both roots found."),
                _fb(3, "(1,-1) is one correct solution pair.",
                    missing="A quadratic gives two x values. The second root x=4 gives y=4-2=2, so the second point is (4,2).",
                    deduction="0.5 mark deducted because only one of the two solution pairs was given.",
                    improve="After finding both x values, always substitute EACH one back to find its matching y value."),
            ],
            topic="simultaneous_linear_quadratic", difficulty="difficult",
        ),
        _example(
            "Find the sum of the first 8 terms of the AP 3, 7, 11, ...", "arithmetic series", 1.5, 2.0,
            steps=[
                _step(1, "[2(3) + 7(4)] = 34", "correct", 1.0),
                _step(2, "34", "partial", 0.5, "Forgot to multiply by n/2 — should be 4 x 34 = 136"),
            ],
            scheme=[
                _scheme(1, "34", 1.0, "Evaluate the bracket [2a+(n-1)d]"),
                _scheme(2, "136", 1.0, "Multiply by n/2"),
            ],
            feedback=[
                _fb(1, "Correctly evaluated the bracket [2a+(n-1)d]."),
                _fb(2, "You correctly evaluated the bracket.",
                    missing="The sum formula is Sn=(n/2)[2a+(n-1)d] — the n/2 factor (here 8/2=4) was never applied: S8=4×34=136.",
                    deduction="0.5 mark deducted because the final multiplication by n/2 was omitted.",
                    improve="Always write out the full sum formula and apply every part of it, including the n/2 factor."),
            ],
            topic="arithmetic_sequences", difficulty="medium",
        ),
        _example(
            "Find the common ratio and 5th term of the GP 4, 8, 16, ...", "geometric sequence", 1.0, 2.0,
            steps=[
                _step(1, "r = 2", "correct", 1.0),
                _step(2, "4 x 2^4 = 4 x 8 = 32", "incorrect", 0.0, "2^4 = 16, not 8"),
            ],
            scheme=[
                _scheme(1, "r = 2", 1.0, "Find the common ratio"),
                _scheme(2, "64", 1.0, "T_n = a r^(n-1), with a=4, r=2"),
            ],
            feedback=[
                _fb(1, "Correctly identified the common ratio."),
                _fb(2, "You correctly set up the nth term formula.",
                    missing="2^4 = 16, not 8 — this looks like an arithmetic slip in computing the power.",
                    deduction="Marks lost because the wrong power value led to an incorrect final answer (should be 64, not 32).",
                    improve="Compute powers step by step to avoid slips: 2^1=2, 2^2=4, 2^3=8, 2^4=16."),
            ],
            topic="geometric_sequences", difficulty="medium",
        ),
        _example(
            "Find the equation of the line through (0, 4) and (2, 0)", "straight line equation", 1.5, 2.0,
            steps=[
                _step(1, "gradient = (0-4)/(2-0) = -2", "correct", 1.0),
                _step(2, "y - 4 = -2(x - 0)", "partial", 0.5,
                      "Not simplified to final y=mx+c form — should be y = -2x + 4"),
            ],
            scheme=[
                _scheme(1, "gradient = -2", 1.0, "Apply (y2-y1)/(x2-x1)"),
                _scheme(2, "y = -2x + 4", 1.0, "Simplify to y=mx+c form (note (0,4) is already the y-intercept)"),
            ],
            feedback=[
                _fb(1, "Correctly calculated the gradient."),
                _fb(2, "Correct use of point-gradient form.",
                    missing="This must be simplified to final y=mx+c form: y-4=-2x, so y=-2x+4. Note (0,4) is already the y-intercept, so c=4 could have been read off directly.",
                    deduction="0.5 mark deducted because the equation was not simplified to its final form.",
                    improve="Always finish by rearranging into y=mx+c form, and remember a point with x=0 directly gives the y-intercept."),
            ],
            topic="straight_line_equations", difficulty="easy",
        ),
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Partial credit: one partial-heavy example per remaining under-represented
# topic, to keep pushing the corpus-wide validity balance toward target while
# broadening depth across all 39 topics.
# ─────────────────────────────────────────────────────────────────────────────

def _partial_credit_examples() -> List[Dict]:
    return [
        _example(
            "Simplify 8y - 3(y + 4)", "simplifying expressions", 1.5, 2.0,
            steps=[
                _step(1, "8y - 3y - 12", "correct", 1.0),
                _step(2, "8y - 3y - 12", "partial", 0.5, "Not combined — should be simplified to 5y - 12"),
            ],
            scheme=[
                _scheme(1, "8y - 3y - 12", 1.0, "Distribute the -3 across the bracket"),
                _scheme(2, "5y - 12", 1.0, "Combine the y terms"),
            ],
            feedback=[
                _fb(1, "Correctly distributed the -3 across the bracket."),
                _fb(2, "Correct distribution.",
                    missing="8y and -3y are like terms and must be combined: 8y-3y=5y.",
                    deduction="0.5 mark deducted because the expression was left unsimplified.",
                    improve="After distributing, always finish by combining any remaining like terms."),
            ],
            topic="simplifying_expressions", difficulty="easy",
        ),
        _example(
            "Simplify 6c - 2d - c + 5d", "collecting like terms", 1.5, 2.0,
            steps=[
                _step(1, "6c - c = 5c", "correct", 1.0),
                _step(2, "-2d + 5d", "partial", 0.5, "Not combined — should be simplified to 3d"),
            ],
            scheme=[
                _scheme(1, "5c", 1.0, "Combine the c terms"),
                _scheme(2, "3d", 1.0, "Combine the d terms"),
            ],
            feedback=[
                _fb(1, "Correctly combined the c terms."),
                _fb(2, "You correctly identified this step involves combining the d terms.",
                    missing="-2d + 5d must be evaluated: 3d.",
                    deduction="0.5 mark deducted because the arithmetic was left unevaluated.",
                    improve="Always finish arithmetic to a final simplified number."),
            ],
            topic="collecting_like_terms", difficulty="easy",
        ),
        _example(
            "Expand (3x - 2)(x + 4)", "expansion", 1.5, 2.0,
            steps=[_step(1, "3x^2 + 12x - 2x - 8", "partial", 1.5,
                          "All four terms correct but never combined — 12x and -2x should be simplified to 10x")],
            scheme=[_scheme(1, "3x^2 + 10x - 8", 2.0, "Expand using FOIL, then combine like terms")],
            feedback=[
                _fb(1, "All four expanded terms are individually correct.",
                    missing="12x and -2x are like terms and must be combined: 12x-2x=10x, giving 3x^2+10x-8.",
                    deduction="0.5 mark deducted because the expression was left unsimplified.",
                    improve="After expanding with FOIL, always finish by combining any like terms."),
            ],
            topic="expanding_brackets", difficulty="medium",
        ),
        _example(
            "Factorise 10a^2b - 15ab^2", "common factor", 1.0, 2.0,
            steps=[_step(1, "5a(2ab - 3b^2)", "partial", 1.0,
                          "b is also a common factor — the full common factor is 5ab, not just 5a")],
            scheme=[_scheme(1, "5ab(2a - 3b)", 2.0, "Factor out the highest common factor, 5ab")],
            feedback=[
                _fb(1, "You correctly identified 5a as a common factor.",
                    missing="b is also common to both terms (10a^2b and 15ab^2 both contain b). The full highest common factor is 5ab, giving 5ab(2a-3b).",
                    deduction="1 mark deducted because the factorisation is incomplete.",
                    improve="Always check EVERY variable for a common factor, not just some of them. Verify: 5ab(2a-3b)=10a^2b-15ab^2 ✓."),
            ],
            topic="factorising_common_factor", difficulty="medium",
        ),
        _example(
            "Factorise x^2 - 3x - 10", "factorisation", 1.0, 2.0,
            steps=[_step(1, "x = 5 or x = -2", "partial", 1.0,
                          "The question asks to FACTORISE the expression, not solve an equation — the bracket form was never written")],
            scheme=[_scheme(1, "(x - 5)(x + 2)", 2.0, "Find a factor pair of -10 that sums to -3: -5 and 2")],
            feedback=[
                _fb(1, "Your roots x=5 and x=-2 correctly imply the factor pair -5 and 2 (product -10, sum -3).",
                    missing="The question asks to factorise, which means writing the expression in bracket form: (x-5)(x+2), not stating x-values as if solving an equation.",
                    deduction="1 mark deducted because the actual factorised form was never written.",
                    improve="Factorising and solving are related but different tasks — factorising ends with a product of brackets, not a list of x-values."),
            ],
            topic="factorising_quadratic", difficulty="medium",
        ),
        _example(
            "Factorise 2x^2 - 50", "common factor (then difference of squares)", 1.5, 2.0,
            steps=[
                _step(1, "2(x^2 - 25)", "correct", 1.0),
                _step(2, "2(x^2 - 25)", "partial", 0.5, "x^2-25 is itself a difference of squares and factorises further"),
            ],
            scheme=[
                _scheme(1, "2(x^2 - 25)", 1.0, "Factor out the common factor 2"),
                _scheme(2, "2(x - 5)(x + 5)", 1.0, "Factorise the remaining difference of squares"),
            ],
            feedback=[
                _fb(1, "Correctly factored out the common factor 2."),
                _fb(2, "Correctly factored out the common factor 2.",
                    missing="x^2-25 is a difference of squares and factorises further into (x-5)(x+5).",
                    deduction="0.5 mark deducted because the factorisation was not taken to its fully factored form.",
                    improve="After factoring out a common term, always check whether what remains can be factorised further."),
            ],
            topic="difference_of_squares", difficulty="medium",
        ),
        _example(
            "Factorise 27x^3 + 1", "sum of cubes", 1.5, 2.0,
            steps=[
                _step(1, "a = 3x, b = 1", "correct", 1.0),
                _step(2, "(3x + 1)((3x)^2 - 3x + 1)", "partial", 0.5, "(3x)^2 not evaluated — should be simplified to 9x^2"),
            ],
            scheme=[
                _scheme(1, "a=3x, b=1", 1.0, "Identify a and b for the identity a^3+b^3"),
                _scheme(2, "(3x+1)(9x^2-3x+1)", 1.0, "Apply a^3+b^3=(a+b)(a^2-ab+b^2)"),
            ],
            feedback=[
                _fb(1, "Correctly identified a=3x and b=1, since (3x)^3=27x^3."),
                _fb(2, "Correctly applied the sum-of-cubes identity structure.",
                    missing="(3x)^2 must be evaluated: 9x^2, not left as (3x)^2.",
                    deduction="0.5 mark deducted because the squared term was left unevaluated.",
                    improve="Always finish arithmetic to a final simplified number or expression."),
            ],
            topic="sum_difference_of_cubes", difficulty="difficult",
        ),
        _example(
            "Solve 6x - 5 = 2x + 11", "linear equation", 1.5, 2.0,
            steps=[
                _step(1, "4x = 16", "correct", 1.0),
                _step(2, "x = 16/4", "partial", 0.5, "Not simplified — should be x = 4"),
            ],
            scheme=[
                _scheme(1, "4x = 16", 1.0, "Collect x terms and constants on opposite sides"),
                _scheme(2, "x = 4", 1.0, "Divide both sides by 4"),
            ],
            feedback=[
                _fb(1, "Correctly collected terms."),
                _fb(2, "Correct method of dividing by 4.",
                    missing="16/4 must be evaluated to a final number: x=4.",
                    deduction="0.5 mark deducted because the fraction was left unsimplified.",
                    improve="Always finish arithmetic to a final simplified number."),
            ],
            topic="linear_equations", difficulty="easy",
        ),
        _example(
            "Solve 5(x + 1) - 3 = 17", "linear equation with brackets", 1.5, 2.0,
            steps=[
                _step(1, "5x + 5 - 3 = 17 → 5x + 2 = 17 → 5x = 15", "correct", 1.0),
                _step(2, "x = 15/5", "partial", 0.5, "Not simplified — should be x = 3"),
            ],
            scheme=[
                _scheme(1, "5x = 15", 1.0, "Distribute and combine constants"),
                _scheme(2, "x = 3", 1.0, "Divide both sides by 5"),
            ],
            feedback=[
                _fb(1, "Correctly distributed and combined the constants."),
                _fb(2, "Correct method of dividing by 5.",
                    missing="15/5 must be evaluated to a final number: x=3.",
                    deduction="0.5 mark deducted because the fraction was left unsimplified.",
                    improve="Always finish arithmetic to a final simplified number."),
            ],
            topic="equations_with_brackets", difficulty="easy",
        ),
        _example(
            "Solve (3x)/4 - 1 = 5", "linear equation with fractions", 1.5, 2.0,
            steps=[
                _step(1, "3x/4 = 6 → 3x = 24", "correct", 1.0),
                _step(2, "x = 24/3", "partial", 0.5, "Not simplified — should be x = 8"),
            ],
            scheme=[
                _scheme(1, "3x = 24", 1.0, "Add 1, then multiply both sides by 4"),
                _scheme(2, "x = 8", 1.0, "Divide both sides by 3"),
            ],
            feedback=[
                _fb(1, "Correctly cleared the fraction."),
                _fb(2, "Correct method of dividing by 3.",
                    missing="24/3 must be evaluated to a final number: x=8.",
                    deduction="0.5 mark deducted because the fraction was left unsimplified.",
                    improve="Always finish arithmetic to a final simplified number."),
            ],
            topic="equations_with_fractions", difficulty="medium",
        ),
        _example(
            "Solve: y = x + 6, 3x + 2y = 2", "substitution", 1.5, 2.0,
            steps=[
                _step(1, "3x + 2(x+6) = 2 → 5x + 12 = 2 → 5x = -10", "correct", 1.0),
                _step(2, "x = -10/5", "partial", 0.5, "Not simplified — should be x = -2, then y = 4"),
            ],
            scheme=[
                _scheme(1, "5x = -10", 1.0, "Substitute y=x+6"),
                _scheme(2, "x = -2, y = 4", 1.0, "Solve for x, then find y"),
            ],
            feedback=[
                _fb(1, "Correctly substituted and simplified."),
                _fb(2, "Correct method of dividing by 5.",
                    missing="-10/5 must be evaluated: x=-2, then y=x+6=4.",
                    deduction="0.5 mark deducted because the answer was left unsimplified and incomplete.",
                    improve="Always finish arithmetic to a final number, and remember to find BOTH variables."),
            ],
            topic="simultaneous_substitution", difficulty="medium",
        ),
        _example(
            "Solve: 2x + 3y = 13, 2x - y = 1", "elimination", 1.5, 2.0,
            steps=[
                _step(1, "Subtracting: 4y = 12 → y = 3", "correct", 1.0),
                _step(2, "y = 3", "partial", 0.5, "x was never found — substitute back: 2x+3(3)=13 gives x=2"),
            ],
            scheme=[
                _scheme(1, "y = 3", 1.0, "Subtract the equations to eliminate x"),
                _scheme(2, "x = 2, y = 3", 1.0, "Substitute back to find x"),
            ],
            feedback=[
                _fb(1, "Correctly eliminated x by subtracting."),
                _fb(2, "y=3 is correct.",
                    missing="The question requires BOTH x and y. Substitute y=3 back into either equation: 2x+3(3)=13 gives x=2.",
                    deduction="0.5 mark deducted because the solution is incomplete — x was never found.",
                    improve="After finding one variable, always substitute back to find the other before finishing."),
            ],
            topic="simultaneous_elimination", difficulty="medium",
        ),
        _example(
            "Solve: y = 2x + 1, y = x^2 + x - 1", "substitution (linear-quadratic)", 2.5, 3.0,
            steps=[
                _step(1, "x^2 - x - 2 = 0", "correct", 1.0),
                _step(2, "(x - 2)(x + 1) = 0 → x = 2 or x = -1", "correct", 1.0),
                _step(3, "y = 2(2) + 1 = 5", "partial", 0.5, "Only one solution pair found — the second point (-1, -1) is missing"),
            ],
            scheme=[
                _scheme(1, "x^2-x-2=0", 1.0, "Substitute to eliminate y"),
                _scheme(2, "x=2 or x=-1", 1.0, "Factorise and solve"),
                _scheme(3, "(2,5) and (-1,-1)", 1.0, "State BOTH solution pairs"),
            ],
            feedback=[
                _fb(1, "Correct substitution and rearrangement."),
                _fb(2, "Correct factorisation and both roots found."),
                _fb(3, "(2,5) is one correct solution pair.",
                    missing="The second root x=-1 gives y=2(-1)+1=-1, so the second point is (-1,-1).",
                    deduction="0.5 mark deducted because only one of the two solution pairs was given.",
                    improve="After finding both x values, always substitute EACH one back to find its matching y value."),
            ],
            topic="simultaneous_linear_quadratic", difficulty="difficult",
        ),
        _example(
            "Solve x^2 + 4x - 21 = 0", "factorisation", 1.5, 2.0,
            steps=[
                _step(1, "(x + 7)(x - 3) = 0", "correct", 1.0),
                _step(2, "x = 3", "partial", 0.5, "Missing the second root x = -7"),
            ],
            scheme=[
                _scheme(1, "(x+7)(x-3)=0", 1.0, "Find a factor pair of -21 that sums to 4: 7 and -3"),
                _scheme(2, "x=-7 or x=3", 1.0, "Apply the zero-product property to BOTH factors"),
            ],
            feedback=[
                _fb(1, "Correct factorisation — 7 x -3 = -21 and 7 + -3 = 4."),
                _fb(2, "x=3 is one correct root, from (x-3)=0.",
                    missing="The other factor (x+7)=0 gives x=-7, which is also a valid root.",
                    deduction="0.5 mark deducted because only one of the two roots was stated.",
                    improve="After factorising, always apply the zero-product property to EVERY factor, not just one."),
            ],
            topic="quadratic_factorisation", difficulty="medium",
        ),
        _example(
            "Solve x^2 - 6x + 4 = 0 using the quadratic formula", "quadratic formula", 1.5, 2.0,
            steps=[
                _step(1, "x = (6 ± √(36 - 16))/2 = (6 ± √20)/2", "correct", 1.0),
                _step(2, "x = (6 ± √20)/2", "partial", 0.5, "Left unsimplified — √20=2√5, so x should reduce to 3 ± √5"),
            ],
            scheme=[
                _scheme(1, "(6±√20)/2", 1.0, "Apply the quadratic formula with a=1, b=-6, c=4"),
                _scheme(2, "x = 3 ± √5", 1.0, "Simplify √20=2√5 and reduce the fraction"),
            ],
            feedback=[
                _fb(1, "Correctly applied the quadratic formula and computed the discriminant."),
                _fb(2, "Correct value under the square root.",
                    missing="√20 simplifies to 2√5, giving x=(6±2√5)/2 = 3±√5.",
                    deduction="0.5 mark deducted because the surd was left unsimplified.",
                    improve="Always simplify a surd and reduce the fraction to its simplest form before finalising an answer."),
            ],
            topic="quadratic_formula", difficulty="difficult",
        ),
        _example(
            "Solve x^2 + 4x - 3 = 0 by completing the square", "completing the square", 2.0, 3.0,
            steps=[
                _step(1, "(x + 2)^2 - 7 = 0", "correct", 1.0),
                _step(2, "(x + 2)^2 = 7", "correct", 1.0),
                _step(3, "x + 2 = √7", "partial", 0.0, "Missing the negative case and the final subtraction step"),
            ],
            scheme=[
                _scheme(1, "(x+2)^2 - 7 = 0", 1.0, "Complete the square (half of 4 is 2)"),
                _scheme(2, "(x+2)^2 = 7", 1.0, "Simplify constants"),
                _scheme(3, "x = -2 ± √7", 1.0, "Take BOTH square roots, then subtract 2"),
            ],
            feedback=[
                _fb(1, "Correctly completed the square using half of the x-coefficient."),
                _fb(2, "Correctly simplified the constants."),
                _fb(3, "You correctly began taking the square root.",
                    missing="Taking a square root gives two cases (x+2=±√7), and the final step (subtracting 2) was never done: x=-2±√7.",
                    deduction="Marks lost because the solution was left incomplete — no numeric value of x was ever reached.",
                    improve="Always finish by writing x+a=±√k, THEN isolate x completely."),
            ],
            topic="completing_the_square", difficulty="difficult",
        ),
        _example(
            "Simplify (x^2 - 4)/(x^2 - 2x)", "simplifying algebraic fractions", 1.5, 2.0,
            steps=[
                _step(1, "(x - 2)(x + 2) / (x(x - 2))", "correct", 1.0),
                _step(2, "(x + 2)/x", "partial", 0.5, "Missing the restriction x ≠ 0, 2"),
            ],
            scheme=[
                _scheme(1, "(x-2)(x+2)/(x(x-2))", 1.0, "Factorise numerator and denominator"),
                _scheme(2, "(x+2)/x, x≠0,2", 1.0, "Cancel and state both restrictions"),
            ],
            feedback=[
                _fb(1, "Correctly factorised both numerator and denominator."),
                _fb(2, "Correctly cancelled the common factor (x-2).",
                    missing="The restrictions x≠0 and x≠2 must be stated, since the original denominator x(x-2) cannot equal zero.",
                    deduction="0.5 mark deducted for omitting the domain restrictions.",
                    improve="Whenever you cancel a factor from a denominator, state ALL values of x that must be excluded, including any left in the simplified denominator."),
            ],
            topic="algebraic_fractions", difficulty="medium",
        ),
        _example(
            "Simplify (x^4)^3 ÷ x^5", "index laws", 1.5, 2.0,
            steps=[
                _step(1, "(x^4)^3 = x^12", "correct", 1.0),
                _step(2, "x^12 / x^5", "partial", 0.5, "Not simplified — subtract the exponents: 12-5=7"),
            ],
            scheme=[
                _scheme(1, "x^12", 1.0, "Multiply exponents when raising a power to a power"),
                _scheme(2, "x^7", 1.0, "Subtract exponents when dividing"),
            ],
            feedback=[
                _fb(1, "Correctly multiplied the exponents: 4x3=12."),
                _fb(2, "Correct method of dividing powers of the same base.",
                    missing="12-5 must be evaluated: x^7.",
                    deduction="0.5 mark deducted because the subtraction was left incomplete.",
                    improve="Always finish arithmetic to a final simplified exponent."),
            ],
            topic="index_laws", difficulty="medium",
        ),
        _example(
            "Evaluate 4^(-1/2)", "index laws (negative index, no working shown)", 0.0, 2.0,
            steps=[
                _step(1, "4^(-1/2) = 2", "incorrect", 0.0,
                      "The negative sign means take the reciprocal AFTER taking the square root: 4^(1/2)=2, then reciprocate to get 1/2, not 2"),
            ],
            scheme=[_scheme(1, "1/2", 2.0, "Take the square root (1/2 power), then apply the negative sign as a reciprocal")],
            feedback=[
                _fb(1, "You reached a numeric answer, showing some awareness that 4^(1/2)=2 is involved.",
                    missing="The exponent is -1/2, and the negative sign means take the RECIPROCAL: 4^(-1/2)=1/4^(1/2)=1/2, not 2.",
                    deduction="Full marks lost because the negative sign on the exponent was dropped, and no working was shown.",
                    improve="Handle a negative fractional exponent in two visible steps: take the root, THEN reciprocate because of the negative sign — always show both steps."),
            ],
            topic="negative_fractional_indices", difficulty="medium",
        ),
        _example(
            "Simplify √48 + √27", "surds", 1.5, 2.0,
            steps=[
                _step(1, "√48 = 4√3", "correct", 1.0),
                _step(2, "4√3 + √27", "partial", 0.5, "√27 also simplifies to 3√3 — then combine: 4√3+3√3=7√3"),
            ],
            scheme=[
                _scheme(1, "4√3", 1.0, "Simplify √48"),
                _scheme(2, "7√3", 1.0, "Simplify √27, then combine like surds"),
            ],
            feedback=[
                _fb(1, "Correctly simplified √48 to 4√3."),
                _fb(2, "Correctly simplified √48.",
                    missing="√27 also simplifies (√27=√(9x3)=3√3). Once both are in terms of √3, they can be combined: 4√3+3√3=7√3.",
                    deduction="0.5 mark deducted because √27 was left unsimplified, preventing the surds from being combined.",
                    improve="Always simplify EVERY surd in an expression before attempting to add or subtract them."),
            ],
            topic="surds", difficulty="medium",
        ),
        _example(
            "Rationalise 2/(3√2)", "surds (rationalisation)", 1.5, 2.0,
            steps=[
                _step(1, "2/(3√2) × √2/√2 = 2√2/6", "correct", 1.0),
                _step(2, "2√2/6", "partial", 0.5, "Not reduced to simplest form — should be √2/3"),
            ],
            scheme=[
                _scheme(1, "2√2/6", 1.0, "Multiply top and bottom by √2"),
                _scheme(2, "√2/3", 1.0, "Reduce the fraction to simplest form"),
            ],
            feedback=[
                _fb(1, "Correctly rationalised the denominator."),
                _fb(2, "Correct rationalisation.",
                    missing="2√2/6 can be reduced further — both 2 and 6 share a factor of 2, giving √2/3.",
                    deduction="0.5 mark deducted because the fraction was left unreduced.",
                    improve="Always check whether a fraction can be simplified further after rationalising."),
            ],
            topic="rationalising_denominators", difficulty="medium",
        ),
        _example(
            "If f(x) = x^2 - 3x, find the values of x for which f(x) = 0", "function evaluation (roots)", 1.5, 2.0,
            steps=[
                _step(1, "x(x - 3) = 0", "correct", 1.0),
                _step(2, "x = 3", "partial", 0.5, "Missing the second solution x = 0"),
            ],
            scheme=[
                _scheme(1, "x(x-3)=0", 1.0, "Factorise f(x)"),
                _scheme(2, "x=0 or x=3", 1.0, "Apply the zero-product property to BOTH factors"),
            ],
            feedback=[
                _fb(1, "Correct factorisation."),
                _fb(2, "x=3 is one correct root, from (x-3)=0.",
                    missing="The factor x=0 also gives a valid root — the other solution is x=0.",
                    deduction="0.5 mark deducted because only one of the two roots was stated.",
                    improve="After factorising, apply the zero-product property to EVERY factor, including a bare x."),
            ],
            topic="functions", difficulty="medium",
        ),
        _example(
            "If f(x) = 3x - 2 and g(x) = x + 5, find (f∘g)(1)", "function composition", 1.5, 2.0,
            steps=[
                _step(1, "g(1) = 6", "correct", 1.0),
                _step(2, "f(6)", "partial", 0.5, "Not evaluated — f(6)=3(6)-2=16"),
            ],
            scheme=[
                _scheme(1, "g(1)=6", 1.0, "Evaluate the inner function"),
                _scheme(2, "f(6)=16", 1.0, "Evaluate f at the result"),
            ],
            feedback=[
                _fb(1, "Correctly evaluated the inner function g(1)."),
                _fb(2, "Correct method of applying f to the result.",
                    missing="f(6) must be evaluated: 3(6)-2=16.",
                    deduction="0.5 mark deducted because the final evaluation was never carried out.",
                    improve="Always finish by evaluating the outer function numerically."),
            ],
            topic="function_composition", difficulty="medium",
        ),
        _example(
            "Find the inverse of f(x) = 5 - x", "inverse function", 1.5, 2.0,
            steps=[
                _step(1, "x = 5 - y", "correct", 1.0),
                _step(2, "x = 5 - y", "partial", 0.5, "Never rearranged for y — should be y = 5 - x"),
            ],
            scheme=[
                _scheme(1, "x = 5 - y", 1.0, "Swap x and y"),
                _scheme(2, "y = 5 - x", 1.0, "Rearrange to solve for y"),
            ],
            feedback=[
                _fb(1, "Correctly swapped x and y."),
                _fb(2, "Correct swap.",
                    missing="The equation must be rearranged to isolate y: y=5-x (this function happens to be its own inverse).",
                    deduction="0.5 mark deducted because the final rearrangement step was never completed.",
                    improve="Always finish by fully isolating y on one side."),
            ],
            topic="inverse_functions", difficulty="medium",
        ),
        _example(
            "Find the equation of the line with gradient -3 passing through (1, 2)", "straight line equation", 1.5, 2.0,
            steps=[
                _step(1, "y - 2 = -3(x - 1)", "correct", 1.0),
                _step(2, "y - 2 = -3(x - 1)", "partial", 0.5, "Not simplified to y=mx+c form — should be y = -3x + 5"),
            ],
            scheme=[
                _scheme(1, "y-2=-3(x-1)", 1.0, "Apply point-gradient form"),
                _scheme(2, "y = -3x + 5", 1.0, "Simplify to y=mx+c form"),
            ],
            feedback=[
                _fb(1, "Correctly applied point-gradient form."),
                _fb(2, "Correct setup.",
                    missing="This must be expanded and rearranged: y-2=-3x+3, so y=-3x+5.",
                    deduction="0.5 mark deducted because the equation was not simplified to its final form.",
                    improve="Always finish by rearranging into y=mx+c form."),
            ],
            topic="straight_line_equations", difficulty="easy",
        ),
        _example(
            "Find the x-intercept and y-intercept of 3x + 2y = 12", "gradient and intercept", 1.0, 2.0,
            steps=[_step(1, "x-intercept: x = 4", "partial", 1.0, "Only the x-intercept was found — the y-intercept is also required")],
            scheme=[_scheme(1, "x-intercept (4,0), y-intercept (0,6)", 2.0, "Set y=0 for the x-intercept, and x=0 for the y-intercept")],
            feedback=[
                _fb(1, "Correctly found the x-intercept by setting y=0.",
                    missing="The y-intercept is also required: setting x=0 gives 2y=12, so y=6.",
                    deduction="1 mark deducted because only one of the two required intercepts was found.",
                    improve="When asked for both intercepts, set y=0 for the x-intercept AND x=0 for the y-intercept."),
            ],
            topic="gradient_intercept", difficulty="easy",
        ),
        _example(
            "The nth term of a sequence is 2n^2 - 1. Find the 4th term.", "sequences (nth term, evaluation)", 1.0, 2.0,
            steps=[_step(1, "2(4)^2 - 1", "partial", 1.0, "Not evaluated to a final number — should be 31")],
            scheme=[_scheme(1, "31", 2.0, "Substitute n=4 and evaluate")],
            feedback=[
                _fb(1, "Correctly substituted n=4 into the formula.",
                    missing="2(4)^2-1 must be evaluated: 2(16)-1=31.",
                    deduction="1 mark deducted because the arithmetic was left unevaluated.",
                    improve="Always finish arithmetic to a final simplified number."),
            ],
            topic="sequences_nth_term", difficulty="easy",
        ),
        _example(
            "The first term of an AP is 6 and the 5th term is 22. Find the common difference.",
            "arithmetic sequence", 1.5, 2.0,
            steps=[
                _step(1, "6 + 4d = 22 → 4d = 16", "correct", 1.0),
                _step(2, "d = 16/4", "partial", 0.5, "Not simplified — should be d = 4"),
            ],
            scheme=[
                _scheme(1, "4d = 16", 1.0, "Apply the nth term formula for the 5th term"),
                _scheme(2, "d = 4", 1.0, "Divide both sides by 4"),
            ],
            feedback=[
                _fb(1, "Correctly set up the equation using the nth term formula."),
                _fb(2, "Correct method of dividing by 4.",
                    missing="16/4 must be evaluated to a final number: d=4.",
                    deduction="0.5 mark deducted because the fraction was left unsimplified.",
                    improve="Always finish arithmetic to a final simplified number."),
            ],
            topic="arithmetic_sequences", difficulty="medium",
        ),
        _example(
            "The first term of a GP is 5 and r=3. Find the sum of the first 4 terms.",
            "geometric series", 1.5, 2.0,
            steps=[
                _step(1, "5(3^4 - 1)/(3 - 1) = 5(80)/2", "correct", 1.0),
                _step(2, "5(80)/2", "partial", 0.5, "Not evaluated — should be 200"),
            ],
            scheme=[
                _scheme(1, "5(80)/2", 1.0, "Apply the geometric series sum formula with a=5, r=3, n=4"),
                _scheme(2, "200", 1.0, "Evaluate"),
            ],
            feedback=[
                _fb(1, "Correctly applied the geometric series sum formula and computed 3^4-1=80."),
                _fb(2, "Correct setup.",
                    missing="5(80)/2 must be evaluated: 400/2=200.",
                    deduction="0.5 mark deducted because the final arithmetic was left incomplete.",
                    improve="Always finish arithmetic to a final simplified number."),
            ],
            topic="geometric_sequences", difficulty="medium",
        ),
        _example(
            "The length of a rectangle is twice its width. If the area is 50 cm^2, find the width.",
            "algebraic word problem", 1.5, 2.0,
            steps=[
                _step(1, "w x 2w = 50 → 2w^2 = 50 → w^2 = 25", "correct", 1.0),
                _step(2, "w = ±5", "partial", 0.5, "Width must be positive — the negative solution should be rejected"),
            ],
            scheme=[
                _scheme(1, "w^2 = 25", 1.0, "Translate into an equation and simplify"),
                _scheme(2, "w = 5", 1.0, "Take the square root and reject the negative solution (width can't be negative)"),
            ],
            feedback=[
                _fb(1, "Correctly translated the problem and simplified to w^2=25."),
                _fb(2, "Correctly took the square root, finding both ±5.",
                    missing="Since w represents a physical width, it cannot be negative. The negative solution w=-5 must be rejected, leaving w=5.",
                    deduction="0.5 mark deducted because the invalid negative solution was not rejected.",
                    improve="In word problems, always check whether a mathematically valid solution makes sense in context — reject solutions that don't (e.g. negative lengths)."),
            ],
            topic="algebraic_word_problems", difficulty="medium",
        ),
        _example(
            "Divide $80 between A and B in the ratio 3:5", "ratio and proportion", 1.5, 2.0,
            steps=[
                _step(1, "8 parts, each part = 80/8 = 10", "correct", 1.0),
                _step(2, "A = 30", "partial", 0.5, "B's share is also required — B = 50"),
            ],
            scheme=[
                _scheme(1, "each part = 10", 1.0, "Find the value of one part"),
                _scheme(2, "A = 30, B = 50", 1.0, "Find BOTH shares"),
            ],
            feedback=[
                _fb(1, "Correctly found the value of one part."),
                _fb(2, "A=30 is correct.",
                    missing="B's share is also needed: B=5x10=50.",
                    deduction="0.5 mark deducted because only one of the two shares was found.",
                    improve="When dividing an amount between two people, always state BOTH shares."),
            ],
            topic="ratio_proportion", difficulty="easy",
        ),
        _example(
            "Make t the subject of v = u + at", "rearranging formulas", 1.5, 2.0,
            steps=[
                _step(1, "at = v - u", "correct", 1.0),
                _step(2, "at = v - u", "partial", 0.5, "Never divided by a — should be t = (v-u)/a"),
            ],
            scheme=[
                _scheme(1, "at = v - u", 1.0, "Subtract u from both sides"),
                _scheme(2, "t = (v-u)/a", 1.0, "Divide both sides by a"),
            ],
            feedback=[
                _fb(1, "Correctly subtracted u from both sides."),
                _fb(2, "Correct rearrangement so far.",
                    missing="To fully isolate t, divide both sides by a: t=(v-u)/a.",
                    deduction="0.5 mark deducted because the final division step was never completed.",
                    improve="Always finish by fully isolating the target variable — check nothing is still multiplying or dividing it."),
            ],
            topic="rearranging_formulas", difficulty="medium",
        ),
        _example(
            "Solve |4x - 8| = 0", "absolute value equation (single solution)", 2.0, 2.0,
            steps=[_step(1, "4x - 8 = 0 → x = 2", "correct", 2.0)],
            scheme=[_scheme(1, "x = 2", 2.0, "When the right-hand side is 0, there is only one case to solve")],
            feedback=[_fb(1, "Correct — when |expression|=0, there is only one solution since there's no separate negative case (0 and -0 are the same).",
                           improve="Full marks.")],
            topic="absolute_value_equations", difficulty="easy",
        ),
        _example(
            "Solve 9^x = 3^(x+2)", "exponent equation (different bases)", 1.5, 2.0,
            steps=[
                _step(1, "3^(2x) = 3^(x+2) → 2x = x + 2", "correct", 1.0),
                _step(2, "2x = x + 2", "partial", 0.5, "Left unsolved — should continue to x = 2"),
            ],
            scheme=[
                _scheme(1, "2x = x+2", 1.0, "Rewrite 9 as 3^2, then equate exponents"),
                _scheme(2, "x = 2", 1.0, "Solve the resulting equation"),
            ],
            feedback=[
                _fb(1, "Correctly rewrote 9 as 3^2 and equated the exponents."),
                _fb(2, "Correct equation set up.",
                    missing="2x=x+2 must still be solved: subtract x from both sides to get x=2.",
                    deduction="0.5 mark deducted because the equation was never solved for x.",
                    improve="After equating exponents, always finish by solving the resulting equation."),
            ],
            topic="exponent_equations", difficulty="medium",
        ),
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Broad pass across 25 topics, 2 examples each, maintaining the
# converged 30/35/35 validity and 30/45/25 difficulty target ratios.
# ─────────────────────────────────────────────────────────────────────────────

def _broad_topic_pass_examples() -> List[Dict]:
    return [
        _example(
            "Expand (x + 6)(x - 1)", "expansion", 2.0, 2.0,
            steps=[
                _step(1, "x^2 - x + 6x - 6", "correct", 1.0),
                _step(2, "x^2 + 5x - 6", "correct", 1.0),
            ],
            scheme=[
                _scheme(1, "x^2-x+6x-6", 1.0, "Expand using FOIL"),
                _scheme(2, "x^2+5x-6", 1.0, "Combine like terms"),
            ],
            feedback=[
                _fb(1, "Correct expansion using FOIL."),
                _fb(2, "Correctly combined like terms.", improve="Full marks."),
            ],
            topic="expanding_brackets", difficulty="easy",
        ),
        _example(
            "Expand (2x + 1)(3x - 4)", "expansion", 0.0, 2.0,
            steps=[_step(1, "6x^2 - 8x + 3x + 4", "incorrect", 0.0, "(1)(-4)=-4, not +4")],
            scheme=[_scheme(1, "6x^2 - 5x - 4", 2.0, "Expand using FOIL, then combine like terms")],
            feedback=[
                _fb(1, "You correctly expanded the first three terms of the FOIL expansion.",
                    missing="The last term is (1)(-4)=-4, not +4. The full expansion is 6x^2-8x+3x-4=6x^2-5x-4.",
                    deduction="Full marks lost because the sign of the constant term was wrong.",
                    improve="Carefully track the sign of each term when multiplying, especially the last (Last) term in FOIL."),
            ],
            topic="expanding_brackets", difficulty="medium",
        ),
        _example(
            "Factorise 100 - x^2", "difference of squares", 2.0, 2.0,
            steps=[_step(1, "(10 - x)(10 + x)", "correct", 2.0)],
            scheme=[_scheme(1, "(10 - x)(10 + x)", 2.0, "Recognise as a^2-b^2 with a=10, b=x")],
            feedback=[_fb(1, "Correct — 10^2=100.", improve="Full marks.")],
            topic="difference_of_squares", difficulty="easy",
        ),
        _example(
            "Factorise 3x^2 - 27", "common factor + difference of squares (fully factorised)", 2.0, 2.0,
            steps=[
                _step(1, "3(x^2 - 9)", "correct", 1.0),
                _step(2, "3(x - 3)(x + 3)", "correct", 1.0),
            ],
            scheme=[
                _scheme(1, "3(x^2-9)", 1.0, "Factor out the common factor 3"),
                _scheme(2, "3(x-3)(x+3)", 1.0, "Factorise the remaining difference of squares"),
            ],
            feedback=[
                _fb(1, "Correctly factored out the common factor 3."),
                _fb(2, "Correctly factorised the remaining difference of squares.",
                    improve="Full marks — fully factorised. Verify: 3(x-3)(x+3)=3(x^2-9)=3x^2-27 ✓."),
            ],
            topic="difference_of_squares", difficulty="medium",
        ),
        _example(
            "Solve 6(x - 1) = 2(x + 7)", "linear equation with brackets", 2.0, 2.0,
            steps=[
                _step(1, "6x - 6 = 2x + 14", "correct", 1.0),
                _step(2, "4x = 20 → x = 5", "correct", 1.0),
            ],
            scheme=[
                _scheme(1, "6x-6=2x+14", 1.0, "Distribute both brackets fully"),
                _scheme(2, "x=5", 1.0, "Solve for x"),
            ],
            feedback=[
                _fb(1, "Correctly distributed both brackets."),
                _fb(2, "Correctly solved for x.", improve="Full marks. Verify: 6(4)=24, 2(12)=24 ✓."),
            ],
            topic="equations_with_brackets", difficulty="medium",
        ),
        _example(
            "Solve 3(2x + 1) = 15", "linear equation with brackets", 0.0, 2.0,
            steps=[
                _step(1, "6x + 1 = 15", "incorrect", 0.0,
                      "3(2x+1)=6x+3, not 6x+1 — both terms must be multiplied by 3"),
                _step(2, "6x = 14 → x = 7/3", "incorrect", 0.0, "Follows from the distribution error"),
            ],
            scheme=[
                _scheme(1, "6x + 3 = 15", 1.0, "Distribute the 3 fully"),
                _scheme(2, "x = 2", 1.0, "Solve for x"),
            ],
            feedback=[
                _fb(1, "You correctly distributed the 3 across the 2x term.",
                    missing="3(2x+1) means BOTH terms are multiplied by 3: 6x+3, not 6x+1.",
                    deduction="Full marks lost because the bracket was not fully distributed."),
                _fb(2, "Your algebra correctly followed from your Step 1 equation.",
                    missing="Because Step 1 was wrong, this answer is wrong. Solving 6x+3=15 correctly gives x=2.",
                    deduction="Marks lost because this follows from the earlier distribution error.",
                    improve="Always distribute a bracket to EVERY term inside it."),
            ],
            topic="equations_with_brackets", difficulty="medium",
        ),
        _example(
            "Solve (x + 5)/2 = 9", "linear equation with fractions", 2.0, 2.0,
            steps=[
                _step(1, "x + 5 = 18", "correct", 1.0),
                _step(2, "x = 13", "correct", 1.0),
            ],
            scheme=[
                _scheme(1, "x + 5 = 18", 1.0, "Multiply both sides by 2"),
                _scheme(2, "x = 13", 1.0, "Subtract 5 from both sides"),
            ],
            feedback=[
                _fb(1, "Correctly cleared the fraction."),
                _fb(2, "Correctly solved for x.", improve="Full marks."),
            ],
            topic="equations_with_fractions", difficulty="easy",
        ),
        _example(
            "Solve 2x/5 - 1 = 3", "linear equation with fractions", 1.5, 2.0,
            steps=[
                _step(1, "2x/5 = 4 → 2x = 20", "correct", 1.0),
                _step(2, "x = 20/2", "partial", 0.5, "Not simplified — should be x = 10"),
            ],
            scheme=[
                _scheme(1, "2x = 20", 1.0, "Add 1, then multiply both sides by 5"),
                _scheme(2, "x = 10", 1.0, "Divide both sides by 2"),
            ],
            feedback=[
                _fb(1, "Correctly cleared the fraction."),
                _fb(2, "Correct method of dividing by 2.",
                    missing="20/2 must be evaluated to a final number: x=10.",
                    deduction="0.5 mark deducted because the fraction was left unsimplified.",
                    improve="Always finish arithmetic to a final simplified number."),
            ],
            topic="equations_with_fractions", difficulty="medium",
        ),
        _example(
            "Solve: y = x - 4, x + y = 10", "substitution", 2.0, 2.0,
            steps=[
                _step(1, "x + x - 4 = 10 → 2x = 14", "correct", 1.0),
                _step(2, "x = 7, y = 3", "correct", 1.0),
            ],
            scheme=[
                _scheme(1, "2x = 14", 1.0, "Substitute y = x-4"),
                _scheme(2, "x = 7, y = 3", 1.0, "Solve for x, then find y"),
            ],
            feedback=[
                _fb(1, "Correctly substituted y=x-4 into the second equation."),
                _fb(2, "Correctly solved for x and y.", improve="Full marks. Verify: 7+3=10 ✓."),
            ],
            topic="simultaneous_substitution", difficulty="easy",
        ),
        _example(
            "Solve: y = 2x + 3, 4x - y = 1", "substitution", 0.0, 2.0,
            steps=[
                _step(1, "4x - 2x + 3 = 1", "incorrect", 0.0,
                      "-(2x+3) = -2x-3, not -2x+3 — the sign of both terms must flip"),
                _step(2, "2x = -2 → x = -1", "incorrect", 0.0, "Follows from the sign error"),
            ],
            scheme=[
                _scheme(1, "4x - 2x - 3 = 1", 1.0, "Substitute y = 2x+3, distributing the negative sign fully"),
                _scheme(2, "x = 2, y = 7", 1.0, "Solve for x, then find y"),
            ],
            feedback=[
                _fb(1, "You correctly substituted y=2x+3 into the second equation.",
                    missing="-(2x+3) means both terms flip sign: -2x-3, not -2x+3.",
                    deduction="Full marks lost because the bracket's sign was not fully distributed."),
                _fb(2, "Your algebra correctly followed from your Step 1 equation.",
                    missing="Because Step 1 was wrong, this answer is wrong. Solving 4x-2x-3=1 correctly gives x=2, y=7.",
                    deduction="Marks lost because this follows from the earlier sign error.",
                    improve="Always distribute a negative sign to EVERY term inside the substituted expression."),
            ],
            topic="simultaneous_substitution", difficulty="medium",
        ),
        _example(
            "Solve x^2 - 2x - 24 = 0", "factorisation", 2.0, 2.0,
            steps=[
                _step(1, "(x - 6)(x + 4) = 0", "correct", 1.0),
                _step(2, "x = 6 or x = -4", "correct", 1.0),
            ],
            scheme=[
                _scheme(1, "(x-6)(x+4)=0", 1.0, "Find a factor pair of -24 that sums to -2: -6 and 4"),
                _scheme(2, "x=6 or x=-4", 1.0, "Apply the zero-product property"),
            ],
            feedback=[
                _fb(1, "Correct factorisation."),
                _fb(2, "Both roots correctly found.", improve="Full marks."),
            ],
            topic="quadratic_factorisation", difficulty="medium",
        ),
        _example(
            "Solve x^2 + 6x + 8 = 0", "factorisation", 1.5, 2.0,
            steps=[
                _step(1, "(x + 2)(x + 4) = 0", "correct", 1.0),
                _step(2, "x = -2", "partial", 0.5, "Missing the second root x = -4"),
            ],
            scheme=[
                _scheme(1, "(x+2)(x+4)=0", 1.0, "Find a factor pair of 8 that sums to 6: 2 and 4"),
                _scheme(2, "x=-2 or x=-4", 1.0, "Apply the zero-product property to BOTH factors"),
            ],
            feedback=[
                _fb(1, "Correct factorisation."),
                _fb(2, "x=-2 is one correct root.",
                    missing="The other factor (x+4)=0 gives x=-4, which is also a valid root.",
                    deduction="0.5 mark deducted because only one of the two roots was stated.",
                    improve="After factorising, apply the zero-product property to EVERY factor."),
            ],
            topic="quadratic_factorisation", difficulty="medium",
        ),
        _example(
            "Add (5x^2 - x + 3) and (-2x^2 + 4x - 7)", "polynomial addition", 1.0, 1.0,
            steps=[_step(1, "3x^2 + 3x - 4", "correct", 1.0)],
            scheme=[_scheme(1, "3x^2 + 3x - 4", 1.0, "Add corresponding like terms")],
            feedback=[_fb(1, "Correctly combined all like terms.", improve="Full marks.")],
            topic="polynomial_addition_subtraction", difficulty="easy",
        ),
        _example(
            "Subtract (3x - 5) from (x^2 + 2x)", "polynomial subtraction", 0.0, 2.0,
            steps=[
                _step(1, "x^2 + 2x - 3x - 5", "incorrect", 0.0,
                      "-(3x-5) = -3x+5, not -3x-5 — the second term's sign must also flip"),
            ],
            scheme=[_scheme(1, "x^2 - x + 5", 2.0, "Distribute the negative sign to every term")],
            feedback=[
                _fb(1, "You correctly set up the subtraction and flipped the sign of the first term.",
                    missing="-(3x-5) means both terms flip: -3x+5, not -3x-5.",
                    deduction="Full marks lost because the second term's sign was not flipped.",
                    improve="When subtracting a bracket, distribute the negative sign to EVERY term inside it."),
            ],
            topic="polynomial_addition_subtraction", difficulty="medium",
        ),
        _example(
            "Multiply (x - 1)(x + 1)(x + 2)", "polynomial multiplication", 2.0, 2.0,
            steps=[
                _step(1, "(x - 1)(x + 1) = x^2 - 1", "correct", 1.0),
                _step(2, "(x^2 - 1)(x + 2) = x^3 + 2x^2 - x - 2", "correct", 1.0),
            ],
            scheme=[
                _scheme(1, "x^2 - 1", 1.0, "Expand the first two brackets (difference of squares)"),
                _scheme(2, "x^3+2x^2-x-2", 1.0, "Multiply the result by the third factor"),
            ],
            feedback=[
                _fb(1, "Correctly recognised and applied the difference-of-squares pattern."),
                _fb(2, "Correctly multiplied the result by the remaining factor.", improve="Full marks."),
            ],
            topic="polynomial_multiplication", difficulty="difficult",
        ),
        _example(
            "Multiply (2x - 3)(2x + 3)", "polynomial multiplication (difference of squares)", 1.5, 2.0,
            steps=[_step(1, "4x^2 + 6x - 6x - 9", "partial", 1.5,
                          "All four terms correct but never combined — the middle terms cancel to give 4x^2-9")],
            scheme=[_scheme(1, "4x^2 - 9", 2.0, "Expand using FOIL, then combine like terms")],
            feedback=[
                _fb(1, "All four expanded terms are individually correct.",
                    missing="6x and -6x are like terms and cancel to 0, giving 4x^2-9.",
                    deduction="0.5 mark deducted because the expression was left unsimplified.",
                    improve="After expanding, always finish by combining any like terms, even when they cancel to zero."),
            ],
            topic="polynomial_multiplication", difficulty="easy",
        ),
        _example(
            "Divide (10x^3 - 15x^2) by 5x^2", "polynomial division", 2.0, 2.0,
            steps=[
                _step(1, "10x^3/5x^2 - 15x^2/5x^2", "correct", 1.0),
                _step(2, "2x - 3", "correct", 1.0),
            ],
            scheme=[
                _scheme(1, "10x^3/5x^2 - 15x^2/5x^2", 1.0, "Divide each term by 5x^2"),
                _scheme(2, "2x - 3", 1.0, "Simplify each quotient"),
            ],
            feedback=[
                _fb(1, "Correctly divided each term by 5x^2."),
                _fb(2, "Correctly simplified.", improve="Full marks."),
            ],
            topic="polynomial_division", difficulty="easy",
        ),
        _example(
            "Divide (x^2 - 4x - 5) by (x + 1)", "polynomial division (by factorisation)", 0.0, 2.0,
            steps=[_step(1, "(x - 5)(x + 1)/(x + 1) = x + 5", "incorrect", 0.0,
                          "Cancelling (x+1) leaves x-5, not x+5 — sign error")],
            scheme=[_scheme(1, "x - 5", 2.0, "Factorise the dividend, then cancel (x+1)")],
            feedback=[
                _fb(1, "You correctly factorised the dividend as (x-5)(x+1).",
                    missing="Cancelling the common factor (x+1) leaves the OTHER factor, x-5, not x+5.",
                    deduction="Full marks lost for the sign error in the final quotient.",
                    improve="After cancelling a common factor, carefully copy the REMAINING factor exactly as it was written."),
            ],
            topic="polynomial_division", difficulty="medium",
        ),
        _example(
            "Simplify (x^2y^3)(x^4y)", "index laws", 1.0, 1.0,
            steps=[_step(1, "x^6y^4", "correct", 1.0)],
            scheme=[_scheme(1, "x^6y^4", 1.0, "Add exponents of matching bases separately")],
            feedback=[_fb(1, "Correctly added the x exponents (2+4=6) and y exponents (3+1=4) separately.",
                           improve="Full marks.")],
            topic="index_laws", difficulty="easy",
        ),
        _example(
            "Simplify x^6 ÷ x^2 ÷ x", "index laws", 1.0, 2.0,
            steps=[
                _step(1, "x^6/x^2 = x^4", "correct", 1.0),
                _step(2, "x^4/x = x^4", "incorrect", 0.0, "Should subtract 1 from the exponent: x^3, not x^4"),
            ],
            scheme=[
                _scheme(1, "x^4", 1.0, "Subtract exponents: 6-2"),
                _scheme(2, "x^3", 1.0, "Subtract exponents again: 4-1"),
            ],
            feedback=[
                _fb(1, "Correctly subtracted the exponents: 6-2=4."),
                _fb(2, "You correctly recognised another division step is needed.",
                    missing="Dividing by x subtracts 1 from the exponent: x^4/x=x^3, not x^4 (dividing by x is not a no-op).",
                    deduction="Full marks lost because the exponent was not reduced.",
                    improve="Remember x is the same as x^1 — dividing by it still subtracts 1 from the exponent."),
            ],
            topic="index_laws", difficulty="medium",
        ),
        _example(
            "Evaluate 9^(3/2)", "index laws (fractional indices)", 2.0, 2.0,
            steps=[
                _step(1, "√9 = 3", "correct", 1.0),
                _step(2, "3^3 = 27", "correct", 1.0),
            ],
            scheme=[
                _scheme(1, "√9 = 3", 1.0, "Take the square root (denominator of the exponent)"),
                _scheme(2, "27", 1.0, "Raise to the power 3 (numerator of the exponent)"),
            ],
            feedback=[
                _fb(1, "Correctly took the square root of 9."),
                _fb(2, "Correctly cubed the result.", improve="Full marks."),
            ],
            topic="negative_fractional_indices", difficulty="medium",
        ),
        _example(
            "Evaluate 32^(-1/5)", "index laws (negative fractional indices)", 1.0, 2.0,
            steps=[
                _step(1, "32^(1/5) = 2", "correct", 1.0),
                _step(2, "= 2", "partial", 0.0, "Negative exponent means take the reciprocal — should be 1/2"),
            ],
            scheme=[_scheme(1, "1/2", 2.0, "Take the fifth root, then apply the negative sign as a reciprocal")],
            feedback=[
                _fb(1, "Correctly took the fifth root of 32."),
                _fb(2, "You correctly evaluated the fifth root.",
                    missing="The exponent is -1/5, and the negative sign means take the RECIPROCAL: 32^(-1/5)=1/32^(1/5)=1/2, not 2.",
                    deduction="Marks lost because the negative sign on the exponent was dropped.",
                    improve="Handle a negative fractional exponent in two steps: take the root, THEN reciprocate."),
            ],
            topic="negative_fractional_indices", difficulty="difficult",
        ),
        _example(
            "Rationalise 7/√7", "surds (rationalisation)", 2.0, 2.0,
            steps=[
                _step(1, "7/√7 × √7/√7 = 7√7/7", "correct", 1.0),
                _step(2, "= √7", "correct", 1.0),
            ],
            scheme=[
                _scheme(1, "7√7/7", 1.0, "Multiply top and bottom by √7"),
                _scheme(2, "√7", 1.0, "Cancel the common factor of 7"),
            ],
            feedback=[
                _fb(1, "Correctly multiplied top and bottom by √7."),
                _fb(2, "Correctly simplified — the 7s cancel.", improve="Full marks."),
            ],
            topic="rationalising_denominators", difficulty="easy",
        ),
        _example(
            "Rationalise 6/(√3 + 1)", "surds (rationalisation with conjugate)", 0.0, 2.0,
            steps=[_step(1, "6(√3 + 1)/(√3 + 1)^2", "incorrect", 0.0,
                          "Must multiply by the CONJUGATE (opposite sign), not the same expression — this doesn't clear the surd")],
            scheme=[_scheme(1, "3(√3 - 1)", 2.0, "Multiply top and bottom by the conjugate √3-1")],
            feedback=[
                _fb(1, "You correctly recognised that multiplying top and bottom by something is needed.",
                    missing="To clear a surd in a denominator like √3+1, multiply by its CONJUGATE (√3-1), which uses the difference-of-squares identity to eliminate the surd. Multiplying by the same expression (√3+1) again does not clear it.",
                    deduction="Full marks lost because the wrong multiplier was used — the surd remains in the denominator.",
                    improve="For a denominator of the form a+√b, always multiply by its conjugate a-√b (opposite sign), never by the same expression again."),
            ],
            topic="rationalising_denominators", difficulty="difficult",
        ),
        _example(
            "If f(x) = 4x + 1, find x such that f(x) = 21", "function (solve for input)", 2.0, 2.0,
            steps=[
                _step(1, "4x + 1 = 21 → 4x = 20", "correct", 1.0),
                _step(2, "x = 5", "correct", 1.0),
            ],
            scheme=[
                _scheme(1, "4x = 20", 1.0, "Set the function equal to 21 and rearrange"),
                _scheme(2, "x = 5", 1.0, "Solve for x"),
            ],
            feedback=[
                _fb(1, "Correctly rearranged the equation."),
                _fb(2, "Correctly solved for x.", improve="Full marks."),
            ],
            topic="functions", difficulty="easy",
        ),
        _example(
            "If f(x) = 2x^2, find f(-3)", "function evaluation", 0.0, 1.0,
            steps=[_step(1, "f(-3) = 2(-3)^2 = 2(-9) = -18", "incorrect", 0.0, "(-3)^2 = 9, not -9")],
            scheme=[_scheme(1, "18", 1.0, "Substitute x=-3 and evaluate")],
            feedback=[
                _fb(1, "You correctly substituted x=-3 into the function.",
                    missing="(-3)^2 = (-3)x(-3) = 9, a positive number, not -9.",
                    deduction="Full marks lost because squaring a negative number was handled incorrectly.",
                    improve="A negative number squared is always positive. Correct: f(-3)=2(9)=18."),
            ],
            topic="functions", difficulty="easy",
        ),
        _example(
            "The perimeter of a square is 36 cm. Find the side length.", "algebraic word problem", 2.0, 2.0,
            steps=[
                _step(1, "4s = 36", "correct", 1.0),
                _step(2, "s = 9", "correct", 1.0),
            ],
            scheme=[
                _scheme(1, "4s = 36", 1.0, "Set up the perimeter equation"),
                _scheme(2, "s = 9", 1.0, "Solve for s"),
            ],
            feedback=[
                _fb(1, "Correctly set up the perimeter equation for a square."),
                _fb(2, "Correctly solved for the side length.", improve="Full marks."),
            ],
            topic="algebraic_word_problems", difficulty="easy",
        ),
        _example(
            "A number tripled and increased by 4 gives 31. Find the number.", "algebraic word problem", 1.5, 2.0,
            steps=[
                _step(1, "3x + 4 = 31 → 3x = 27", "correct", 1.0),
                _step(2, "x = 27/3", "partial", 0.5, "Not simplified — should be x = 9"),
            ],
            scheme=[
                _scheme(1, "3x = 27", 1.0, "Translate into an equation"),
                _scheme(2, "x = 9", 1.0, "Divide both sides by 3"),
            ],
            feedback=[
                _fb(1, "Correctly translated the problem into an equation."),
                _fb(2, "Correct method of dividing by 3.",
                    missing="27/3 must be evaluated to a final number: x=9.",
                    deduction="0.5 mark deducted because the fraction was left unsimplified.",
                    improve="Always finish arithmetic to a final simplified number."),
            ],
            topic="algebraic_word_problems", difficulty="easy",
        ),
        _example(
            "Make b the subject of P = 2l + 2b", "rearranging formulas", 2.0, 2.0,
            steps=[
                _step(1, "2b = P - 2l", "correct", 1.0),
                _step(2, "b = (P - 2l)/2", "correct", 1.0),
            ],
            scheme=[
                _scheme(1, "2b = P - 2l", 1.0, "Subtract 2l from both sides"),
                _scheme(2, "b = (P-2l)/2", 1.0, "Divide both sides by 2"),
            ],
            feedback=[
                _fb(1, "Correctly subtracted 2l from both sides."),
                _fb(2, "Correctly divided by 2.", improve="Full marks."),
            ],
            topic="rearranging_formulas", difficulty="medium",
        ),
        _example(
            "Make x the subject of y = x^2 + 3", "rearranging formulas", 1.0, 2.0,
            steps=[
                _step(1, "x^2 = y - 3", "correct", 1.0),
                _step(2, "x = y - 3", "incorrect", 0.0, "Forgot to take the square root — should be x = √(y-3)"),
            ],
            scheme=[
                _scheme(1, "x^2 = y - 3", 1.0, "Subtract 3 from both sides"),
                _scheme(2, "x = √(y-3)", 1.0, "Take the square root of both sides"),
            ],
            feedback=[
                _fb(1, "Correctly isolated x^2."),
                _fb(2, "You correctly isolated x^2.",
                    missing="To undo the square on x, take the square root of both sides: x=√(y-3), not just y-3.",
                    deduction="Full marks lost because the square root step was skipped.",
                    improve="Whenever a variable is squared, the inverse operation is a square root — never skip it."),
            ],
            topic="rearranging_formulas", difficulty="medium",
        ),
        _example(
            "Solve 5 - 2x ≤ 11", "linear inequality", 2.0, 2.0,
            steps=[
                _step(1, "5 - 2x ≤ 11 → -2x ≤ 6", "correct", 1.0),
                _step(2, "x ≥ -3", "correct", 1.0),
            ],
            scheme=[
                _scheme(1, "-2x ≤ 6", 1.0, "Subtract 5 from both sides"),
                _scheme(2, "x ≥ -3", 1.0, "Divide by -2 and flip the inequality"),
            ],
            feedback=[
                _fb(1, "Correctly subtracted 5 from both sides."),
                _fb(2, "Correctly divided by -2 and flipped the inequality.", improve="Full marks."),
            ],
            topic="linear_inequalities", difficulty="medium",
        ),
        _example(
            "Solve 3x + 7 > 1", "linear inequality", 0.0, 2.0,
            steps=[
                _step(1, "3x > 1 - 7 = -8", "incorrect", 0.0, "1 - 7 = -6, not -8"),
                _step(2, "x > -8/3", "incorrect", 0.0, "Follows from the arithmetic slip"),
            ],
            scheme=[
                _scheme(1, "3x > -6", 1.0, "Subtract 7 from both sides"),
                _scheme(2, "x > -2", 1.0, "Divide both sides by 3"),
            ],
            feedback=[
                _fb(1, "Correct method of subtracting 7 from both sides.",
                    missing="1 - 7 = -6, not -8 — this looks like an arithmetic slip.",
                    deduction="Full marks lost for the incorrect subtraction."),
                _fb(2, "Your division correctly followed from your Step 1 result.",
                    missing="Because Step 1 was wrong, this answer is wrong. Solving 3x>-6 correctly gives x>-2.",
                    deduction="Marks lost because this follows from the earlier arithmetic slip.",
                    improve="Double-check basic subtraction, especially with negative numbers."),
            ],
            topic="linear_inequalities", difficulty="easy",
        ),
        _example(
            "Solve x^2 - 16 ≤ 0", "quadratic inequality", 2.0, 2.0,
            steps=[
                _step(1, "(x - 4)(x + 4) ≤ 0", "correct", 1.0),
                _step(2, "-4 ≤ x ≤ 4", "correct", 1.0),
            ],
            scheme=[
                _scheme(1, "(x-4)(x+4) ≤ 0", 1.0, "Factorise the quadratic"),
                _scheme(2, "-4 ≤ x ≤ 4", 1.0, "Identify the region between the roots"),
            ],
            feedback=[
                _fb(1, "Correct factorisation."),
                _fb(2, "Correctly identified the region where the product is negative or zero.", improve="Full marks."),
            ],
            topic="quadratic_inequalities", difficulty="medium",
        ),
        _example(
            "Solve x^2 - 2x - 8 < 0", "quadratic inequality", 1.5, 2.0,
            steps=[
                _step(1, "(x - 4)(x + 2) < 0", "correct", 1.0),
                _step(2, "x < 4", "partial", 0.5, "Missing the lower bound — should be -2 < x < 4"),
            ],
            scheme=[
                _scheme(1, "(x-4)(x+2) < 0", 1.0, "Factorise the quadratic"),
                _scheme(2, "-2 < x < 4", 1.0, "Identify the region between the roots"),
            ],
            feedback=[
                _fb(1, "Correct factorisation."),
                _fb(2, "You correctly identified x=4 as an upper limit.",
                    missing="The product of two factors is negative only BETWEEN the roots: -2 < x < 4, not just x < 4.",
                    deduction="0.5 mark deducted for the missing lower bound.",
                    improve="For a quadratic inequality with two real roots, sketch a sign diagram to find the full solution region."),
            ],
            topic="quadratic_inequalities", difficulty="difficult",
        ),
        _example(
            "Solve: y = 3x, y = x^2 + 2x", "substitution (linear-quadratic)", 3.0, 3.0,
            steps=[
                _step(1, "3x = x^2 + 2x → x^2 - x = 0", "correct", 1.0),
                _step(2, "x(x - 1) = 0 → x = 0 or x = 1", "correct", 1.0),
                _step(3, "(0, 0) and (1, 3)", "correct", 1.0),
            ],
            scheme=[
                _scheme(1, "x^2-x=0", 1.0, "Substitute to eliminate y"),
                _scheme(2, "x=0 or x=1", 1.0, "Factorise and solve"),
                _scheme(3, "(0,0) and (1,3)", 1.0, "Substitute back to find both y values"),
            ],
            feedback=[
                _fb(1, "Correct substitution and rearrangement."),
                _fb(2, "Correct factorisation and both roots found."),
                _fb(3, "Both solution pairs correctly found.", improve="Full marks."),
            ],
            topic="simultaneous_linear_quadratic", difficulty="medium",
        ),
        _example(
            "Solve: y = x + 5, y = x^2 - 1", "substitution (linear-quadratic)", 1.0, 2.0,
            steps=[
                _step(1, "x^2 - x - 6 = 0", "correct", 1.0),
                _step(2, "(x + 3)(x - 2) = 0 → x = -3 or x = 2", "incorrect", 0.0,
                      "Wrong factors: should be (x-3)(x+2), since -3x2=-6 but -3+2=-1 does not match; the correct pair is -3 and 2 giving factors (x-3)(x+2)"),
            ],
            scheme=[
                _scheme(1, "x^2-x-6=0", 1.0, "Substitute to eliminate y"),
                _scheme(2, "(x-3)(x+2)=0 → x=3 or x=-2", 1.0, "Find factors of -6 that sum to -1"),
            ],
            feedback=[
                _fb(1, "Correct substitution and rearrangement."),
                _fb(2, "You correctly identified 3 and 2 as relevant numbers (product magnitude 6).",
                    missing="You need factors of -6 that SUM to -1: those are -3 and 2, giving (x-3)(x+2), not (x+3)(x-2) which expands to x^2+x-6.",
                    deduction="Full marks lost because the factorisation does not expand back to x^2-x-6.",
                    improve="Always verify a factorisation by expanding it back out and checking it matches the original expression."),
            ],
            topic="simultaneous_linear_quadratic", difficulty="difficult",
        ),
        _example(
            "Solve 2x^2 + 3x - 2 = 0 using the quadratic formula", "quadratic formula", 2.0, 2.0,
            steps=[
                _step(1, "x = (-3 ± √(9 + 16))/4 = (-3 ± 5)/4", "correct", 1.0),
                _step(2, "x = 1/2 or x = -2", "correct", 1.0),
            ],
            scheme=[
                _scheme(1, "(-3±5)/4", 1.0, "Apply the quadratic formula with a=2, b=3, c=-2"),
                _scheme(2, "x=1/2 or x=-2", 1.0, "Evaluate both cases"),
            ],
            feedback=[
                _fb(1, "Correctly applied the quadratic formula and computed the discriminant."),
                _fb(2, "Correctly evaluated both roots.", improve="Full marks."),
            ],
            topic="quadratic_formula", difficulty="medium",
        ),
        _example(
            "Solve x^2 - 4x + 1 = 0 using the quadratic formula", "quadratic formula", 2.0, 3.0,
            steps=[
                _step(1, "a=1, b=-4, c=1", "correct", 1.0),
                _step(2, "x = (4 ± √(16 - 4))/2 = (4 ± √12)/2", "correct", 1.0),
                _step(3, "x = (4 ± √12)/2", "partial", 0.0,
                      "Not simplified — √12=2√3, giving x = 2 ± √3"),
            ],
            scheme=[
                _scheme(1, "a=1, b=-4, c=1", 1.0, "Identify the coefficients"),
                _scheme(2, "(4±√12)/2", 1.0, "Apply the quadratic formula"),
                _scheme(3, "x = 2 ± √3", 1.0, "Simplify √12=2√3 and reduce the fraction"),
            ],
            feedback=[
                _fb(1, "Correctly identified the coefficients a, b, and c."),
                _fb(2, "Correctly applied the quadratic formula and computed the discriminant."),
                _fb(3, "Correct value under the square root.",
                    missing="√12 simplifies to 2√3, giving x=(4±2√3)/2 = 2±√3.",
                    deduction="Marks lost because the surd was left unsimplified.",
                    improve="Always simplify a surd and reduce the fraction before finalising an answer."),
            ],
            topic="quadratic_formula", difficulty="difficult",
        ),
        _example(
            "Solve x^2 - 10x + 21 = 0 by completing the square", "completing the square", 3.0, 3.0,
            steps=[
                _step(1, "(x - 5)^2 - 4 = 0", "correct", 1.0),
                _step(2, "(x - 5)^2 = 4", "correct", 1.0),
                _step(3, "x = 7 or x = 3", "correct", 1.0),
            ],
            scheme=[
                _scheme(1, "(x-5)^2 - 4 = 0", 1.0, "Complete the square (half of -10 is -5)"),
                _scheme(2, "(x-5)^2 = 4", 1.0, "Simplify constants"),
                _scheme(3, "x = 7 or x = 3", 1.0, "Take the square root and solve"),
            ],
            feedback=[
                _fb(1, "Correctly completed the square."),
                _fb(2, "Correctly simplified the constants."),
                _fb(3, "Both roots correctly stated.", improve="Full marks."),
            ],
            topic="completing_the_square", difficulty="medium",
        ),
        _example(
            "Solve x^2 + 2x - 15 = 0 by completing the square", "completing the square", 1.0, 2.0,
            steps=[
                _step(1, "(x + 1)^2 - 16 = 0", "correct", 1.0),
                _step(2, "(x + 1)^2 = 14", "incorrect", 0.0, "-1 - 15 = -16, not -14"),
            ],
            scheme=[
                _scheme(1, "(x+1)^2 - 16 = 0", 1.0, "Complete the square (half of 2 is 1)"),
                _scheme(2, "x = 3 or x = -5", 1.0, "Simplify, then take both square roots"),
            ],
            feedback=[
                _fb(1, "Correctly completed the square using half of the x-coefficient."),
                _fb(2, "You correctly moved the constants together.",
                    missing="-1 - 15 = -16, not -14 — this looks like an arithmetic slip. The correct equation is (x+1)^2=16.",
                    deduction="Full marks lost because the wrong constant led to wrong final roots.",
                    improve="Double-check basic subtraction with negative numbers. Correct: (x+1)^2=16, so x+1=±4, giving x=3 or x=-5."),
            ],
            topic="completing_the_square", difficulty="difficult",
        ),
        _example(
            "Simplify √12 × √3", "surds", 1.0, 1.0,
            steps=[_step(1, "√36 = 6", "correct", 1.0)],
            scheme=[_scheme(1, "6", 1.0, "Multiply inside the square roots: √12×√3=√36")],
            feedback=[_fb(1, "Correctly multiplied inside the square roots and simplified.", improve="Full marks.")],
            topic="surds", difficulty="easy",
        ),
        _example(
            "Simplify (2√5)^2", "surds", 0.0, 2.0,
            steps=[_step(1, "2 × 5 = 10", "incorrect", 0.0, "The coefficient 2 must also be squared: 2^2×5=20, not 2×5")],
            scheme=[_scheme(1, "20", 2.0, "(2√5)^2 = 2^2 × (√5)^2 = 4 × 5")],
            feedback=[
                _fb(1, "You correctly squared the surd part: (√5)^2=5.",
                    missing="The coefficient 2 must ALSO be squared: 2^2=4, so the full answer is 4x5=20, not 2x5=10.",
                    deduction="Full marks lost because the coefficient was not squared.",
                    improve="When squaring a product like (ab)^2, every factor — including numeric coefficients — must be squared."),
            ],
            topic="surds", difficulty="medium",
        ),
        _example(
            "If f(x) = x - 3 and g(x) = 2x + 1, find (f∘g)(4)", "function composition", 2.0, 2.0,
            steps=[
                _step(1, "g(4) = 9", "correct", 1.0),
                _step(2, "f(9) = 6", "correct", 1.0),
            ],
            scheme=[
                _scheme(1, "g(4) = 9", 1.0, "Evaluate the inner function first"),
                _scheme(2, "f(9) = 6", 1.0, "Apply f to the result"),
            ],
            feedback=[
                _fb(1, "Correctly evaluated the inner function g(4)."),
                _fb(2, "Correctly applied f to the result.", improve="Full marks."),
            ],
            topic="function_composition", difficulty="medium",
        ),
        _example(
            "If f(x) = x^2 - 1 and g(x) = x + 2, find (g∘f)(3)", "function composition", 1.0, 2.0,
            steps=[
                _step(1, "f(3) = 8", "correct", 1.0),
                _step(2, "g(8)", "partial", 0.0, "Not evaluated — g(8)=8+2=10"),
            ],
            scheme=[
                _scheme(1, "f(3) = 8", 1.0, "Evaluate the inner function first"),
                _scheme(2, "g(8) = 10", 1.0, "Apply g to the result"),
            ],
            feedback=[
                _fb(1, "Correctly evaluated the inner function f(3)."),
                _fb(2, "Correct method of applying g to the result.",
                    missing="g(8) must be evaluated: 8+2=10.",
                    deduction="Marks lost because the final evaluation was never carried out.",
                    improve="Always finish by evaluating the outer function numerically."),
            ],
            topic="function_composition", difficulty="medium",
        ),
        _example(
            "Find the inverse of f(x) = 7 - 2x", "inverse function", 2.0, 2.0,
            steps=[
                _step(1, "x = 7 - 2y → 2y = 7 - x", "correct", 1.0),
                _step(2, "y = (7 - x)/2", "correct", 1.0),
            ],
            scheme=[
                _scheme(1, "2y = 7 - x", 1.0, "Swap x and y, then rearrange"),
                _scheme(2, "y = (7-x)/2", 1.0, "Solve for y"),
            ],
            feedback=[
                _fb(1, "Correctly swapped x and y and rearranged."),
                _fb(2, "Correctly solved for y.", improve="Full marks."),
            ],
            topic="inverse_functions", difficulty="medium",
        ),
        _example(
            "Find the inverse of f(x) = (x - 1)/4", "inverse function", 1.0, 2.0,
            steps=[
                _step(1, "4x = y - 1", "correct", 1.0),
                _step(2, "y = 4x - 1", "incorrect", 0.0, "Should add 1 to both sides: y=4x+1, not 4x-1"),
            ],
            scheme=[
                _scheme(1, "4x = y - 1", 1.0, "Swap x and y, then clear the fraction"),
                _scheme(2, "y = 4x + 1", 1.0, "Add 1 to both sides"),
            ],
            feedback=[
                _fb(1, "Correctly swapped x and y and cleared the fraction."),
                _fb(2, "You correctly reached 4x=y-1.",
                    missing="To isolate y, add 1 to both sides: y=4x+1, not 4x-1.",
                    deduction="Full marks lost for the sign error.",
                    improve="Double-check the direction of each operation when isolating a variable."),
            ],
            topic="inverse_functions", difficulty="medium",
        ),
        _example(
            "Find the equation of the line through (-1, 2) and (3, -6)", "straight line equation", 2.0, 2.0,
            steps=[
                _step(1, "gradient = (-6-2)/(3-(-1)) = -2", "correct", 1.0),
                _step(2, "y = -2x", "correct", 1.0),
            ],
            scheme=[
                _scheme(1, "gradient = -2", 1.0, "Apply (y2-y1)/(x2-x1)"),
                _scheme(2, "y = -2x", 1.0, "Substitute a point and the gradient, then simplify"),
            ],
            feedback=[
                _fb(1, "Correctly calculated the gradient."),
                _fb(2, "Correctly formed and simplified the equation.", improve="Full marks. Verify: -2(-1)=2 ✓."),
            ],
            topic="straight_line_equations", difficulty="medium",
        ),
        _example(
            "Find the equation of the line with gradient 5 passing through (-2, -3)", "straight line equation", 1.5, 2.0,
            steps=[
                _step(1, "y + 3 = 5(x + 2)", "correct", 1.0),
                _step(2, "y + 3 = 5(x + 2)", "partial", 0.5, "Not simplified — should be y = 5x + 7"),
            ],
            scheme=[
                _scheme(1, "y+3=5(x+2)", 1.0, "Apply point-gradient form"),
                _scheme(2, "y = 5x + 7", 1.0, "Simplify to y=mx+c form"),
            ],
            feedback=[
                _fb(1, "Correctly applied point-gradient form."),
                _fb(2, "Correct setup.",
                    missing="This must be expanded and rearranged: y+3=5x+10, so y=5x+7.",
                    deduction="0.5 mark deducted because the equation was not simplified to its final form.",
                    improve="Always finish by rearranging into y=mx+c form."),
            ],
            topic="straight_line_equations", difficulty="easy",
        ),
        _example(
            "Find the gradient of a line parallel to y = -x + 8", "gradient and intercept", 1.0, 1.0,
            steps=[_step(1, "gradient = -1", "correct", 1.0)],
            scheme=[_scheme(1, "-1", 1.0, "Parallel lines share the same gradient")],
            feedback=[_fb(1, "Correctly identified that parallel lines share the same gradient.", improve="Full marks.")],
            topic="gradient_intercept", difficulty="easy",
        ),
        _example(
            "Find the y-intercept of 2y = 6x - 10", "gradient and intercept", 0.0, 1.0,
            steps=[_step(1, "2y = 6x - 10 → y-intercept = -10", "incorrect", 0.0,
                          "Must divide EVERY term by 2: y=3x-5, so the y-intercept is -5, not -10")],
            scheme=[_scheme(1, "-5", 1.0, "Divide every term by 2 to reach y=mx+c form first")],
            feedback=[
                _fb(1, "You correctly identified the equation needs to be rearranged into y=mx+c form.",
                    missing="Dividing 2y=6x-10 by 2 gives y=3x-5 — the constant -10 must also be divided by 2, giving a y-intercept of -5, not -10.",
                    deduction="Full marks lost because only part of the equation was divided.",
                    improve="When rearranging to y=mx+c form, divide EVERY term by the coefficient of y, including the constant."),
            ],
            topic="gradient_intercept", difficulty="medium",
        ),
    ]



# ─────────────────────────────────────────────────────────────────────────────
# 221 additional unique examples.
#
# This set is additive only: it preserves every existing example and uses the
# same _example / _step / _scheme / _fb schema.  Together with the existing
# 279 records it produces exactly 500 raw annotations.
#
# Added-record balance:
#   validity   41 correct / 96 partial / 84 incorrect
#   difficulty 55 easy / 87 medium / 79 difficult
# ─────────────────────────────────────────────────────────────────────────────

def _bulk_generated_examples() -> List[Dict]:
    """Additional 221 explicit annotations using the same _example format as earlier sets."""
    return [
        _example(
            'Simplify 5x + 3 - 2x + 4', 'simplifying expressions', 2.0, 2.0,
            steps=[
                _step(1, '3x + 7', 'correct', 2.0),
            ],
            scheme=[
                _scheme(1, '3x + 7', 2.0, 'Complete the algebra accurately and state the final result'),
            ],
            feedback=[
                _fb(1, 'You correctly combined the x terms and the constant terms.',
                    improve='Group like terms first: combine x terms together and constants together.'),
            ],
            topic='simplifying_expressions', difficulty='easy',
        ),
        _example(
            'Simplify 6x + 5 - 3x + 7', 'simplifying expressions', 1.0, 2.0,
            steps=[
                _step(1, '3x + 5 + 7', 'partial', 1.0, 'The x terms were combined, but the constants were left uncombined.'),
            ],
            scheme=[
                _scheme(1, '3x + 12', 2.0, 'Complete the algebra accurately and state the final result'),
            ],
            feedback=[
                _fb(1, 'You made a valid start and used part of the correct algebraic method.',
                    missing='The x terms were combined, but the constants were left uncombined. The complete result is 3x + 12.',
                    deduction='1 mark deducted because the response is incomplete or not fully simplified.',
                    improve='Group like terms first: combine x terms together and constants together.'),
            ],
            topic='simplifying_expressions', difficulty='medium',
        ),
        _example(
            'Simplify 7x + 7 - 4x + 10', 'simplifying expressions', 1.0, 2.0,
            steps=[
                _step(1, '3x + 7 + 10', 'partial', 1.0, 'The x terms were combined, but the constants were left uncombined.'),
            ],
            scheme=[
                _scheme(1, '3x + 17', 2.0, 'Complete the algebra accurately and state the final result'),
            ],
            feedback=[
                _fb(1, 'You made a valid start and used part of the correct algebraic method.',
                    missing='The x terms were combined, but the constants were left uncombined. The complete result is 3x + 17.',
                    deduction='1 mark deducted because the response is incomplete or not fully simplified.',
                    improve='Group like terms first: combine x terms together and constants together.'),
            ],
            topic='simplifying_expressions', difficulty='easy',
        ),
        _example(
            'Simplify 8x + 9 - 5x + 13', 'simplifying expressions', 0.0, 2.0,
            steps=[
                _step(1, '25x', 'incorrect', 0.0, 'Constants and x terms are unlike terms and cannot all be combined as one x term.'),
            ],
            scheme=[
                _scheme(1, '3x + 22', 2.0, 'Complete the algebra accurately and state the final result'),
            ],
            feedback=[
                _fb(1, 'You identified the relevant algebra topic and attempted a solution.',
                    missing='Constants and x terms are unlike terms and cannot all be combined as one x term. The correct result is 3x + 22.',
                    deduction='Full marks lost because the stated result does not follow the required algebraic rule.',
                    improve='Group like terms first: combine x terms together and constants together.'),
            ],
            topic='simplifying_expressions', difficulty='medium',
        ),
        _example(
            'Simplify 9x + 11 - 6x + 16', 'simplifying expressions', 0.0, 2.0,
            steps=[
                _step(1, '30x', 'incorrect', 0.0, 'Constants and x terms are unlike terms and cannot all be combined as one x term.'),
            ],
            scheme=[
                _scheme(1, '3x + 27', 2.0, 'Complete the algebra accurately and state the final result'),
            ],
            feedback=[
                _fb(1, 'You identified the relevant algebra topic and attempted a solution.',
                    missing='Constants and x terms are unlike terms and cannot all be combined as one x term. The correct result is 3x + 27.',
                    deduction='Full marks lost because the stated result does not follow the required algebraic rule.',
                    improve='Group like terms first: combine x terms together and constants together.'),
            ],
            topic='simplifying_expressions', difficulty='difficult',
        ),
        _example(
            'Collect like terms: 4x^2 + 5x - 1x^2 + 3x - 1', 'collecting like terms', 2.0, 2.0,
            steps=[
                _step(1, '3x^2 + 8x - 1', 'correct', 2.0),
            ],
            scheme=[
                _scheme(1, '3x^2 + 8x - 1', 2.0, 'Complete the algebra accurately and state the final result'),
            ],
            feedback=[
                _fb(1, 'You correctly grouped terms with the same variable and exponent.',
                    improve='Only combine terms whose variables and exponents match exactly, and retain unmatched terms.'),
            ],
            topic='collecting_like_terms', difficulty='easy',
        ),
        _example(
            'Collect like terms: 5x^2 + 7x - 2x^2 + 4x - 3', 'collecting like terms', 1.0, 2.0,
            steps=[
                _step(1, '3x^2 + 11x', 'partial', 1.0, 'The constant term was omitted from the final expression.'),
            ],
            scheme=[
                _scheme(1, '3x^2 + 11x - 3', 2.0, 'Complete the algebra accurately and state the final result'),
            ],
            feedback=[
                _fb(1, 'You made a valid start and used part of the correct algebraic method.',
                    missing='The constant term was omitted from the final expression. The complete result is 3x^2 + 11x - 3.',
                    deduction='1 mark deducted because the response is incomplete or not fully simplified.',
                    improve='Only combine terms whose variables and exponents match exactly, and retain unmatched terms.'),
            ],
            topic='collecting_like_terms', difficulty='medium',
        ),
        _example(
            'Collect like terms: 6x^2 + 9x - 3x^2 + 5x - 5', 'collecting like terms', 1.0, 2.0,
            steps=[
                _step(1, '3x^2 + 14x', 'partial', 1.0, 'The constant term was omitted from the final expression.'),
            ],
            scheme=[
                _scheme(1, '3x^2 + 14x - 5', 2.0, 'Complete the algebra accurately and state the final result'),
            ],
            feedback=[
                _fb(1, 'You made a valid start and used part of the correct algebraic method.',
                    missing='The constant term was omitted from the final expression. The complete result is 3x^2 + 14x - 5.',
                    deduction='1 mark deducted because the response is incomplete or not fully simplified.',
                    improve='Only combine terms whose variables and exponents match exactly, and retain unmatched terms.'),
            ],
            topic='collecting_like_terms', difficulty='difficult',
        ),
        _example(
            'Collect like terms: 7x^2 + 11x - 4x^2 + 6x - 7', 'collecting like terms', 0.0, 2.0,
            steps=[
                _step(1, '13x^2', 'incorrect', 0.0, 'Terms containing x^2, x, and no variable are unlike terms and must remain separate.'),
            ],
            scheme=[
                _scheme(1, '3x^2 + 17x - 7', 2.0, 'Complete the algebra accurately and state the final result'),
            ],
            feedback=[
                _fb(1, 'You identified the relevant algebra topic and attempted a solution.',
                    missing='Terms containing x^2, x, and no variable are unlike terms and must remain separate. The correct result is 3x^2 + 17x - 7.',
                    deduction='Full marks lost because the stated result does not follow the required algebraic rule.',
                    improve='Only combine terms whose variables and exponents match exactly, and retain unmatched terms.'),
            ],
            topic='collecting_like_terms', difficulty='medium',
        ),
        _example(
            'Collect like terms: 8x^2 + 13x - 5x^2 + 7x - 9', 'collecting like terms', 0.0, 2.0,
            steps=[
                _step(1, '14x^2', 'incorrect', 0.0, 'Terms containing x^2, x, and no variable are unlike terms and must remain separate.'),
            ],
            scheme=[
                _scheme(1, '3x^2 + 20x - 9', 2.0, 'Complete the algebra accurately and state the final result'),
            ],
            feedback=[
                _fb(1, 'You identified the relevant algebra topic and attempted a solution.',
                    missing='Terms containing x^2, x, and no variable are unlike terms and must remain separate. The correct result is 3x^2 + 20x - 9.',
                    deduction='Full marks lost because the stated result does not follow the required algebraic rule.',
                    improve='Only combine terms whose variables and exponents match exactly, and retain unmatched terms.'),
            ],
            topic='collecting_like_terms', difficulty='difficult',
        ),
        _example(
            'Expand and simplify (x + 2)(x - 5)', 'expanding brackets', 2.0, 2.0,
            steps=[
                _step(1, 'x^2 - 3x - 10', 'correct', 2.0),
            ],
            scheme=[
                _scheme(1, 'x^2 - 3x - 10', 2.0, 'Complete the algebra accurately and state the final result'),
            ],
            feedback=[
                _fb(1, 'You correctly multiplied every term and combined the two middle terms.',
                    improve='Use FOIL or full distribution, then collect the two middle x terms.'),
            ],
            topic='expanding_brackets', difficulty='easy',
        ),
        _example(
            'Expand and simplify (x + 3)(x - 6)', 'expanding brackets', 1.0, 2.0,
            steps=[
                _step(1, 'x^2 - 6x + 3x - 18', 'partial', 1.0, 'All four products are present, but the two x terms were not collected.'),
            ],
            scheme=[
                _scheme(1, 'x^2 - 3x - 18', 2.0, 'Complete the algebra accurately and state the final result'),
            ],
            feedback=[
                _fb(1, 'You made a valid start and used part of the correct algebraic method.',
                    missing='All four products are present, but the two x terms were not collected. The complete result is x^2 - 3x - 18.',
                    deduction='1 mark deducted because the response is incomplete or not fully simplified.',
                    improve='Use FOIL or full distribution, then collect the two middle x terms.'),
            ],
            topic='expanding_brackets', difficulty='medium',
        ),
        _example(
            'Expand and simplify (x + 4)(x - 7)', 'expanding brackets', 1.0, 2.0,
            steps=[
                _step(1, 'x^2 - 7x + 4x - 28', 'partial', 1.0, 'All four products are present, but the two x terms were not collected.'),
            ],
            scheme=[
                _scheme(1, 'x^2 - 3x - 28', 2.0, 'Complete the algebra accurately and state the final result'),
            ],
            feedback=[
                _fb(1, 'You made a valid start and used part of the correct algebraic method.',
                    missing='All four products are present, but the two x terms were not collected. The complete result is x^2 - 3x - 28.',
                    deduction='1 mark deducted because the response is incomplete or not fully simplified.',
                    improve='Use FOIL or full distribution, then collect the two middle x terms.'),
            ],
            topic='expanding_brackets', difficulty='difficult',
        ),
        _example(
            'Expand and simplify (x + 5)(x - 8)', 'expanding brackets', 0.0, 2.0,
            steps=[
                _step(1, 'x^2 - 40', 'incorrect', 0.0, 'The two middle cross-products cannot be omitted when multiplying non-conjugate brackets.'),
            ],
            scheme=[
                _scheme(1, 'x^2 - 3x - 40', 2.0, 'Complete the algebra accurately and state the final result'),
            ],
            feedback=[
                _fb(1, 'You identified the relevant algebra topic and attempted a solution.',
                    missing='The two middle cross-products cannot be omitted when multiplying non-conjugate brackets. The correct result is x^2 - 3x - 40.',
                    deduction='Full marks lost because the stated result does not follow the required algebraic rule.',
                    improve='Use FOIL or full distribution, then collect the two middle x terms.'),
            ],
            topic='expanding_brackets', difficulty='medium',
        ),
        _example(
            'Expand and simplify (x + 6)(x - 9)', 'expanding brackets', 0.0, 2.0,
            steps=[
                _step(1, 'x^2 - 54', 'incorrect', 0.0, 'The two middle cross-products cannot be omitted when multiplying non-conjugate brackets.'),
            ],
            scheme=[
                _scheme(1, 'x^2 - 3x - 54', 2.0, 'Complete the algebra accurately and state the final result'),
            ],
            feedback=[
                _fb(1, 'You identified the relevant algebra topic and attempted a solution.',
                    missing='The two middle cross-products cannot be omitted when multiplying non-conjugate brackets. The correct result is x^2 - 3x - 54.',
                    deduction='Full marks lost because the stated result does not follow the required algebraic rule.',
                    improve='Use FOIL or full distribution, then collect the two middle x terms.'),
            ],
            topic='expanding_brackets', difficulty='difficult',
        ),
        _example(
            'Factorise fully 6x^2 + 10x', 'highest common factor', 2.0, 2.0,
            steps=[
                _step(1, '2x(3x + 5)', 'correct', 2.0),
            ],
            scheme=[
                _scheme(1, '2x(3x + 5)', 2.0, 'Complete the algebra accurately and state the final result'),
            ],
            feedback=[
                _fb(1, 'You correctly extracted the greatest numerical and variable common factor.',
                    improve='Find the greatest common factor of both coefficients and variables, then divide each term by it.'),
            ],
            topic='factorising_common_factor', difficulty='easy',
        ),
        _example(
            'Factorise fully 12x^2 + 21x', 'highest common factor', 1.0, 2.0,
            steps=[
                _step(1, '3(4x^2 + 7x)', 'partial', 1.0, 'The numerical factor was removed, but the common factor x was not extracted.'),
            ],
            scheme=[
                _scheme(1, '3x(4x + 7)', 2.0, 'Complete the algebra accurately and state the final result'),
            ],
            feedback=[
                _fb(1, 'You made a valid start and used part of the correct algebraic method.',
                    missing='The numerical factor was removed, but the common factor x was not extracted. The complete result is 3x(4x + 7).',
                    deduction='1 mark deducted because the response is incomplete or not fully simplified.',
                    improve='Find the greatest common factor of both coefficients and variables, then divide each term by it.'),
            ],
            topic='factorising_common_factor', difficulty='medium',
        ),
        _example(
            'Factorise fully 20x^2 + 36x', 'highest common factor', 1.0, 2.0,
            steps=[
                _step(1, '4(5x^2 + 9x)', 'partial', 1.0, 'The numerical factor was removed, but the common factor x was not extracted.'),
            ],
            scheme=[
                _scheme(1, '4x(5x + 9)', 2.0, 'Complete the algebra accurately and state the final result'),
            ],
            feedback=[
                _fb(1, 'You made a valid start and used part of the correct algebraic method.',
                    missing='The numerical factor was removed, but the common factor x was not extracted. The complete result is 4x(5x + 9).',
                    deduction='1 mark deducted because the response is incomplete or not fully simplified.',
                    improve='Find the greatest common factor of both coefficients and variables, then divide each term by it.'),
            ],
            topic='factorising_common_factor', difficulty='difficult',
        ),
        _example(
            'Factorise fully 30x^2 + 55x', 'highest common factor', 0.0, 2.0,
            steps=[
                _step(1, '5x(6x + 55)', 'incorrect', 0.0, 'After taking out the common factor, every term inside the bracket must be divided by it.'),
            ],
            scheme=[
                _scheme(1, '5x(6x + 11)', 2.0, 'Complete the algebra accurately and state the final result'),
            ],
            feedback=[
                _fb(1, 'You identified the relevant algebra topic and attempted a solution.',
                    missing='After taking out the common factor, every term inside the bracket must be divided by it. The correct result is 5x(6x + 11).',
                    deduction='Full marks lost because the stated result does not follow the required algebraic rule.',
                    improve='Find the greatest common factor of both coefficients and variables, then divide each term by it.'),
            ],
            topic='factorising_common_factor', difficulty='medium',
        ),
        _example(
            'Factorise fully 42x^2 + 78x', 'highest common factor', 0.0, 2.0,
            steps=[
                _step(1, '6x(7x + 78)', 'incorrect', 0.0, 'After taking out the common factor, every term inside the bracket must be divided by it.'),
            ],
            scheme=[
                _scheme(1, '6x(7x + 13)', 2.0, 'Complete the algebra accurately and state the final result'),
            ],
            feedback=[
                _fb(1, 'You identified the relevant algebra topic and attempted a solution.',
                    missing='After taking out the common factor, every term inside the bracket must be divided by it. The correct result is 6x(7x + 13).',
                    deduction='Full marks lost because the stated result does not follow the required algebraic rule.',
                    improve='Find the greatest common factor of both coefficients and variables, then divide each term by it.'),
            ],
            topic='factorising_common_factor', difficulty='difficult',
        ),
        _example(
            'Factorise x^2 + 6x + 8', 'quadratic factorisation', 2.0, 2.0,
            steps=[
                _step(1, '(x + 2)(x + 4)', 'correct', 2.0),
            ],
            scheme=[
                _scheme(1, '(x + 2)(x + 4)', 2.0, 'Complete the algebra accurately and state the final result'),
            ],
            feedback=[
                _fb(1, 'You selected the factor pair whose product is the constant and whose sum is the x coefficient.',
                    improve='Check both conditions: the factors must multiply to the constant and add to the middle coefficient.'),
            ],
            topic='factorising_quadratic', difficulty='easy',
        ),
        _example(
            'Factorise x^2 + 8x + 15', 'quadratic factorisation', 1.0, 2.0,
            steps=[
                _step(1, 'The required factor pair is 3 and 5', 'partial', 1.0, 'The correct factor pair was found, but it was not written as a factorised expression.'),
            ],
            scheme=[
                _scheme(1, '(x + 3)(x + 5)', 2.0, 'Complete the algebra accurately and state the final result'),
            ],
            feedback=[
                _fb(1, 'You made a valid start and used part of the correct algebraic method.',
                    missing='The correct factor pair was found, but it was not written as a factorised expression. The complete result is (x + 3)(x + 5).',
                    deduction='1 mark deducted because the response is incomplete or not fully simplified.',
                    improve='Check both conditions: the factors must multiply to the constant and add to the middle coefficient.'),
            ],
            topic='factorising_quadratic', difficulty='medium',
        ),
        _example(
            'Factorise x^2 + 10x + 24', 'quadratic factorisation', 1.0, 2.0,
            steps=[
                _step(1, 'The required factor pair is 4 and 6', 'partial', 1.0, 'The correct factor pair was found, but it was not written as a factorised expression.'),
            ],
            scheme=[
                _scheme(1, '(x + 4)(x + 6)', 2.0, 'Complete the algebra accurately and state the final result'),
            ],
            feedback=[
                _fb(1, 'You made a valid start and used part of the correct algebraic method.',
                    missing='The correct factor pair was found, but it was not written as a factorised expression. The complete result is (x + 4)(x + 6).',
                    deduction='1 mark deducted because the response is incomplete or not fully simplified.',
                    improve='Check both conditions: the factors must multiply to the constant and add to the middle coefficient.'),
            ],
            topic='factorising_quadratic', difficulty='difficult',
        ),
        _example(
            'Factorise x^2 + 12x + 35', 'quadratic factorisation', 0.0, 2.0,
            steps=[
                _step(1, '(x - 5)(x - 7)', 'incorrect', 0.0, 'Both signs were reversed, so the expanded middle term and constant do not match the question.'),
            ],
            scheme=[
                _scheme(1, '(x + 5)(x + 7)', 2.0, 'Complete the algebra accurately and state the final result'),
            ],
            feedback=[
                _fb(1, 'You identified the relevant algebra topic and attempted a solution.',
                    missing='Both signs were reversed, so the expanded middle term and constant do not match the question. The correct result is (x + 5)(x + 7).',
                    deduction='Full marks lost because the stated result does not follow the required algebraic rule.',
                    improve='Check both conditions: the factors must multiply to the constant and add to the middle coefficient.'),
            ],
            topic='factorising_quadratic', difficulty='medium',
        ),
        _example(
            'Factorise x^2 + 14x + 48', 'quadratic factorisation', 0.0, 2.0,
            steps=[
                _step(1, '(x - 6)(x - 8)', 'incorrect', 0.0, 'Both signs were reversed, so the expanded middle term and constant do not match the question.'),
            ],
            scheme=[
                _scheme(1, '(x + 6)(x + 8)', 2.0, 'Complete the algebra accurately and state the final result'),
            ],
            feedback=[
                _fb(1, 'You identified the relevant algebra topic and attempted a solution.',
                    missing='Both signs were reversed, so the expanded middle term and constant do not match the question. The correct result is (x + 6)(x + 8).',
                    deduction='Full marks lost because the stated result does not follow the required algebraic rule.',
                    improve='Check both conditions: the factors must multiply to the constant and add to the middle coefficient.'),
            ],
            topic='factorising_quadratic', difficulty='difficult',
        ),
        _example(
            'Factorise 4x^2 - 9', 'difference of squares', 2.0, 2.0,
            steps=[
                _step(1, '(2x - 3)(2x + 3)', 'correct', 2.0),
            ],
            scheme=[
                _scheme(1, '(2x - 3)(2x + 3)', 2.0, 'Complete the algebra accurately and state the final result'),
            ],
            feedback=[
                _fb(1, 'You correctly used A^2-B^2=(A-B)(A+B).',
                    improve='Identify A and B, then write one factor with a minus sign and the other with a plus sign.'),
            ],
            topic='difference_of_squares', difficulty='easy',
        ),
        _example(
            'Factorise 9x^2 - 16', 'difference of squares', 1.0, 2.0,
            steps=[
                _step(1, '(3x - 4)(...)', 'partial', 1.0, 'Only one of the two conjugate factors was completed.'),
            ],
            scheme=[
                _scheme(1, '(3x - 4)(3x + 4)', 2.0, 'Complete the algebra accurately and state the final result'),
            ],
            feedback=[
                _fb(1, 'You made a valid start and used part of the correct algebraic method.',
                    missing='Only one of the two conjugate factors was completed. The complete result is (3x - 4)(3x + 4).',
                    deduction='1 mark deducted because the response is incomplete or not fully simplified.',
                    improve='Identify A and B, then write one factor with a minus sign and the other with a plus sign.'),
            ],
            topic='difference_of_squares', difficulty='medium',
        ),
        _example(
            'Factorise 16x^2 - 25', 'difference of squares', 1.0, 2.0,
            steps=[
                _step(1, '(4x - 5)(...)', 'partial', 1.0, 'Only one of the two conjugate factors was completed.'),
            ],
            scheme=[
                _scheme(1, '(4x - 5)(4x + 5)', 2.0, 'Complete the algebra accurately and state the final result'),
            ],
            feedback=[
                _fb(1, 'You made a valid start and used part of the correct algebraic method.',
                    missing='Only one of the two conjugate factors was completed. The complete result is (4x - 5)(4x + 5).',
                    deduction='1 mark deducted because the response is incomplete or not fully simplified.',
                    improve='Identify A and B, then write one factor with a minus sign and the other with a plus sign.'),
            ],
            topic='difference_of_squares', difficulty='difficult',
        ),
        _example(
            'Factorise 25x^2 - 36', 'difference of squares', 0.0, 2.0,
            steps=[
                _step(1, '(5x - 6)^2', 'incorrect', 0.0, 'A difference of squares produces conjugate factors, not two identical factors.'),
            ],
            scheme=[
                _scheme(1, '(5x - 6)(5x + 6)', 2.0, 'Complete the algebra accurately and state the final result'),
            ],
            feedback=[
                _fb(1, 'You identified the relevant algebra topic and attempted a solution.',
                    missing='A difference of squares produces conjugate factors, not two identical factors. The correct result is (5x - 6)(5x + 6).',
                    deduction='Full marks lost because the stated result does not follow the required algebraic rule.',
                    improve='Identify A and B, then write one factor with a minus sign and the other with a plus sign.'),
            ],
            topic='difference_of_squares', difficulty='medium',
        ),
        _example(
            'Factorise 36x^2 - 49', 'difference of squares', 0.0, 2.0,
            steps=[
                _step(1, '(6x - 7)^2', 'incorrect', 0.0, 'A difference of squares produces conjugate factors, not two identical factors.'),
            ],
            scheme=[
                _scheme(1, '(6x - 7)(6x + 7)', 2.0, 'Complete the algebra accurately and state the final result'),
            ],
            feedback=[
                _fb(1, 'You identified the relevant algebra topic and attempted a solution.',
                    missing='A difference of squares produces conjugate factors, not two identical factors. The correct result is (6x - 7)(6x + 7).',
                    deduction='Full marks lost because the stated result does not follow the required algebraic rule.',
                    improve='Identify A and B, then write one factor with a minus sign and the other with a plus sign.'),
            ],
            topic='difference_of_squares', difficulty='difficult',
        ),
        _example(
            'Factorise 1x^3 - 8', 'difference-of-cubes identity', 2.0, 2.0,
            steps=[
                _step(1, '(1x - 2)(1x^2 + 2x + 4)', 'correct', 2.0),
            ],
            scheme=[
                _scheme(1, '(1x - 2)(1x^2 + 2x + 4)', 2.0, 'Complete the algebra accurately and state the final result'),
            ],
            feedback=[
                _fb(1, 'You correctly applied the cube-factorisation identity and computed all three terms.',
                    improve='Write the complete identity first, substitute A and B, and check the middle product AB carefully.'),
            ],
            topic='sum_difference_of_cubes', difficulty='easy',
        ),
        _example(
            'Factorise 8x^3 + 27', 'sum-of-cubes identity', 1.0, 2.0,
            steps=[
                _step(1, '(2x + 3)(4x^2 + ... + 9)', 'partial', 1.0, 'The outer factor and square terms are correct, but the middle ab term is missing.'),
            ],
            scheme=[
                _scheme(1, '(2x + 3)(4x^2 - 6x + 9)', 2.0, 'Complete the algebra accurately and state the final result'),
            ],
            feedback=[
                _fb(1, 'You made a valid start and used part of the correct algebraic method.',
                    missing='The outer factor and square terms are correct, but the middle ab term is missing. The complete result is (2x + 3)(4x^2 - 6x + 9).',
                    deduction='1 mark deducted because the response is incomplete or not fully simplified.',
                    improve='Write the complete identity first, substitute A and B, and check the middle product AB carefully.'),
            ],
            topic='sum_difference_of_cubes', difficulty='medium',
        ),
        _example(
            'Factorise 27x^3 - 64', 'difference-of-cubes identity', 1.0, 2.0,
            steps=[
                _step(1, '(3x - 4)(9x^2 + ... + 16)', 'partial', 1.0, 'The outer factor and square terms are correct, but the middle ab term is missing.'),
            ],
            scheme=[
                _scheme(1, '(3x - 4)(9x^2 + 12x + 16)', 2.0, 'Complete the algebra accurately and state the final result'),
            ],
            feedback=[
                _fb(1, 'You made a valid start and used part of the correct algebraic method.',
                    missing='The outer factor and square terms are correct, but the middle ab term is missing. The complete result is (3x - 4)(9x^2 + 12x + 16).',
                    deduction='1 mark deducted because the response is incomplete or not fully simplified.',
                    improve='Write the complete identity first, substitute A and B, and check the middle product AB carefully.'),
            ],
            topic='sum_difference_of_cubes', difficulty='difficult',
        ),
        _example(
            'Factorise 64x^3 + 125', 'sum-of-cubes identity', 0.0, 2.0,
            steps=[
                _step(1, '(4x + 5)(16x^2 + 20x + 25)', 'incorrect', 0.0, "The sign of the middle ab term must follow the cube identity's alternating-sign pattern."),
            ],
            scheme=[
                _scheme(1, '(4x + 5)(16x^2 - 20x + 25)', 2.0, 'Complete the algebra accurately and state the final result'),
            ],
            feedback=[
                _fb(1, 'You identified the relevant algebra topic and attempted a solution.',
                    missing="The sign of the middle ab term must follow the cube identity's alternating-sign pattern. The correct result is (4x + 5)(16x^2 - 20x + 25).",
                    deduction='Full marks lost because the stated result does not follow the required algebraic rule.',
                    improve='Write the complete identity first, substitute A and B, and check the middle product AB carefully.'),
            ],
            topic='sum_difference_of_cubes', difficulty='medium',
        ),
        _example(
            'Factorise 125x^3 - 216', 'difference-of-cubes identity', 0.0, 2.0,
            steps=[
                _step(1, '(5x - 6)(25x^2 - 30x + 36)', 'incorrect', 0.0, "The sign of the middle ab term must follow the cube identity's alternating-sign pattern."),
            ],
            scheme=[
                _scheme(1, '(5x - 6)(25x^2 + 30x + 36)', 2.0, 'Complete the algebra accurately and state the final result'),
            ],
            feedback=[
                _fb(1, 'You identified the relevant algebra topic and attempted a solution.',
                    missing="The sign of the middle ab term must follow the cube identity's alternating-sign pattern. The correct result is (5x - 6)(25x^2 + 30x + 36).",
                    deduction='Full marks lost because the stated result does not follow the required algebraic rule.',
                    improve='Write the complete identity first, substitute A and B, and check the middle product AB carefully.'),
            ],
            topic='sum_difference_of_cubes', difficulty='difficult',
        ),
        _example(
            'Solve 3x + 5 = 17', 'linear equation', 2.0, 2.0,
            steps=[
                _step(1, 'x = 4', 'correct', 2.0),
            ],
            scheme=[
                _scheme(1, 'x = 4', 2.0, 'Complete the algebra accurately and state the final result'),
            ],
            feedback=[
                _fb(1, 'You correctly isolated x and divided by its coefficient.',
                    improve='Undo addition or subtraction first, then divide both sides by the coefficient of x.'),
            ],
            topic='linear_equations', difficulty='easy',
        ),
        _example(
            'Solve 4x + 7 = 27', 'linear equation', 1.0, 2.0,
            steps=[
                _step(1, '4x = 20', 'partial', 1.0, 'The x term was isolated correctly, but the final division was not completed.'),
            ],
            scheme=[
                _scheme(1, 'x = 5', 2.0, 'Complete the algebra accurately and state the final result'),
            ],
            feedback=[
                _fb(1, 'You made a valid start and used part of the correct algebraic method.',
                    missing='The x term was isolated correctly, but the final division was not completed. The complete result is x = 5.',
                    deduction='1 mark deducted because the response is incomplete or not fully simplified.',
                    improve='Undo addition or subtraction first, then divide both sides by the coefficient of x.'),
            ],
            topic='linear_equations', difficulty='medium',
        ),
        _example(
            'Solve 5x + 9 = 39', 'linear equation', 1.0, 2.0,
            steps=[
                _step(1, '5x = 30', 'partial', 1.0, 'The x term was isolated correctly, but the final division was not completed.'),
            ],
            scheme=[
                _scheme(1, 'x = 6', 2.0, 'Complete the algebra accurately and state the final result'),
            ],
            feedback=[
                _fb(1, 'You made a valid start and used part of the correct algebraic method.',
                    missing='The x term was isolated correctly, but the final division was not completed. The complete result is x = 6.',
                    deduction='1 mark deducted because the response is incomplete or not fully simplified.',
                    improve='Undo addition or subtraction first, then divide both sides by the coefficient of x.'),
            ],
            topic='linear_equations', difficulty='difficult',
        ),
        _example(
            'Solve 6x + 11 = 53', 'linear equation', 0.0, 2.0,
            steps=[
                _step(1, 'x = 10', 'incorrect', 0.0, 'The constant should be subtracted from both sides before dividing, not added.'),
            ],
            scheme=[
                _scheme(1, 'x = 7', 2.0, 'Complete the algebra accurately and state the final result'),
            ],
            feedback=[
                _fb(1, 'You identified the relevant algebra topic and attempted a solution.',
                    missing='The constant should be subtracted from both sides before dividing, not added. The correct result is x = 7.',
                    deduction='Full marks lost because the stated result does not follow the required algebraic rule.',
                    improve='Undo addition or subtraction first, then divide both sides by the coefficient of x.'),
            ],
            topic='linear_equations', difficulty='medium',
        ),
        _example(
            'Solve 7x + 13 = 69', 'linear equation', 0.0, 2.0,
            steps=[
                _step(1, 'x = 11', 'incorrect', 0.0, 'The constant should be subtracted from both sides before dividing, not added.'),
            ],
            scheme=[
                _scheme(1, 'x = 8', 2.0, 'Complete the algebra accurately and state the final result'),
            ],
            feedback=[
                _fb(1, 'You identified the relevant algebra topic and attempted a solution.',
                    missing='The constant should be subtracted from both sides before dividing, not added. The correct result is x = 8.',
                    deduction='Full marks lost because the stated result does not follow the required algebraic rule.',
                    improve='Undo addition or subtraction first, then divide both sides by the coefficient of x.'),
            ],
            topic='linear_equations', difficulty='difficult',
        ),
        _example(
            'Solve 2(x + 1) = 8', 'linear equation with brackets', 2.0, 2.0,
            steps=[
                _step(1, 'x = 3', 'correct', 2.0),
            ],
            scheme=[
                _scheme(1, 'x = 3', 2.0, 'Complete the algebra accurately and state the final result'),
            ],
            feedback=[
                _fb(1, 'You correctly expanded the bracket and solved the resulting linear equation.',
                    improve='Distribute to every term, isolate the x term, and complete the final division.'),
            ],
            topic='equations_with_brackets', difficulty='easy',
        ),
        _example(
            'Solve 3(x + 2) = 18', 'linear equation with brackets', 1.0, 2.0,
            steps=[
                _step(1, '3x = 12', 'partial', 1.0, 'The bracket was expanded and x was isolated, but division by the coefficient was not completed.'),
            ],
            scheme=[
                _scheme(1, 'x = 4', 2.0, 'Complete the algebra accurately and state the final result'),
            ],
            feedback=[
                _fb(1, 'You made a valid start and used part of the correct algebraic method.',
                    missing='The bracket was expanded and x was isolated, but division by the coefficient was not completed. The complete result is x = 4.',
                    deduction='1 mark deducted because the response is incomplete or not fully simplified.',
                    improve='Distribute to every term, isolate the x term, and complete the final division.'),
            ],
            topic='equations_with_brackets', difficulty='medium',
        ),
        _example(
            'Solve 4(x + 3) = 32', 'linear equation with brackets', 1.0, 2.0,
            steps=[
                _step(1, '4x = 20', 'partial', 1.0, 'The bracket was expanded and x was isolated, but division by the coefficient was not completed.'),
            ],
            scheme=[
                _scheme(1, 'x = 5', 2.0, 'Complete the algebra accurately and state the final result'),
            ],
            feedback=[
                _fb(1, 'You made a valid start and used part of the correct algebraic method.',
                    missing='The bracket was expanded and x was isolated, but division by the coefficient was not completed. The complete result is x = 5.',
                    deduction='1 mark deducted because the response is incomplete or not fully simplified.',
                    improve='Distribute to every term, isolate the x term, and complete the final division.'),
            ],
            topic='equations_with_brackets', difficulty='difficult',
        ),
        _example(
            'Solve 5(x + 4) = 50', 'linear equation with brackets', 0.0, 2.0,
            steps=[
                _step(1, 'x = 30', 'incorrect', 0.0, 'After expansion, the isolated x term still has a coefficient and must be divided by that coefficient.'),
            ],
            scheme=[
                _scheme(1, 'x = 6', 2.0, 'Complete the algebra accurately and state the final result'),
            ],
            feedback=[
                _fb(1, 'You identified the relevant algebra topic and attempted a solution.',
                    missing='After expansion, the isolated x term still has a coefficient and must be divided by that coefficient. The correct result is x = 6.',
                    deduction='Full marks lost because the stated result does not follow the required algebraic rule.',
                    improve='Distribute to every term, isolate the x term, and complete the final division.'),
            ],
            topic='equations_with_brackets', difficulty='medium',
        ),
        _example(
            'Solve 6(x + 5) = 72', 'linear equation with brackets', 0.0, 2.0,
            steps=[
                _step(1, 'x = 42', 'incorrect', 0.0, 'After expansion, the isolated x term still has a coefficient and must be divided by that coefficient.'),
            ],
            scheme=[
                _scheme(1, 'x = 7', 2.0, 'Complete the algebra accurately and state the final result'),
            ],
            feedback=[
                _fb(1, 'You identified the relevant algebra topic and attempted a solution.',
                    missing='After expansion, the isolated x term still has a coefficient and must be divided by that coefficient. The correct result is x = 7.',
                    deduction='Full marks lost because the stated result does not follow the required algebraic rule.',
                    improve='Distribute to every term, isolate the x term, and complete the final division.'),
            ],
            topic='equations_with_brackets', difficulty='difficult',
        ),
        _example(
            'Solve (x + 2)/3 = 4', 'linear equation with fractions', 2.0, 2.0,
            steps=[
                _step(1, 'x = 10', 'correct', 2.0),
            ],
            scheme=[
                _scheme(1, 'x = 10', 2.0, 'Complete the algebra accurately and state the final result'),
            ],
            feedback=[
                _fb(1, 'You correctly cleared the denominator and isolated x.',
                    improve='Multiply both sides by the denominator, then use the inverse operation to isolate x.'),
            ],
            topic='equations_with_fractions', difficulty='easy',
        ),
        _example(
            'Solve (x + 3)/4 = 5', 'linear equation with fractions', 1.0, 2.0,
            steps=[
                _step(1, 'x + 3 = 20', 'partial', 1.0, 'The fraction was cleared correctly, but the constant was not moved to finish solving for x.'),
            ],
            scheme=[
                _scheme(1, 'x = 17', 2.0, 'Complete the algebra accurately and state the final result'),
            ],
            feedback=[
                _fb(1, 'You made a valid start and used part of the correct algebraic method.',
                    missing='The fraction was cleared correctly, but the constant was not moved to finish solving for x. The complete result is x = 17.',
                    deduction='1 mark deducted because the response is incomplete or not fully simplified.',
                    improve='Multiply both sides by the denominator, then use the inverse operation to isolate x.'),
            ],
            topic='equations_with_fractions', difficulty='medium',
        ),
        _example(
            'Solve (x + 4)/5 = 6', 'linear equation with fractions', 1.0, 2.0,
            steps=[
                _step(1, 'x + 4 = 30', 'partial', 1.0, 'The fraction was cleared correctly, but the constant was not moved to finish solving for x.'),
            ],
            scheme=[
                _scheme(1, 'x = 26', 2.0, 'Complete the algebra accurately and state the final result'),
            ],
            feedback=[
                _fb(1, 'You made a valid start and used part of the correct algebraic method.',
                    missing='The fraction was cleared correctly, but the constant was not moved to finish solving for x. The complete result is x = 26.',
                    deduction='1 mark deducted because the response is incomplete or not fully simplified.',
                    improve='Multiply both sides by the denominator, then use the inverse operation to isolate x.'),
            ],
            topic='equations_with_fractions', difficulty='difficult',
        ),
        _example(
            'Solve (x + 5)/6 = 7', 'linear equation with fractions', 0.0, 2.0,
            steps=[
                _step(1, 'x = 47', 'incorrect', 0.0, 'After clearing the denominator, the added constant must be subtracted, not added again.'),
            ],
            scheme=[
                _scheme(1, 'x = 37', 2.0, 'Complete the algebra accurately and state the final result'),
            ],
            feedback=[
                _fb(1, 'You identified the relevant algebra topic and attempted a solution.',
                    missing='After clearing the denominator, the added constant must be subtracted, not added again. The correct result is x = 37.',
                    deduction='Full marks lost because the stated result does not follow the required algebraic rule.',
                    improve='Multiply both sides by the denominator, then use the inverse operation to isolate x.'),
            ],
            topic='equations_with_fractions', difficulty='medium',
        ),
        _example(
            'Solve (x + 6)/7 = 8', 'linear equation with fractions', 0.0, 2.0,
            steps=[
                _step(1, 'x = 62', 'incorrect', 0.0, 'After clearing the denominator, the added constant must be subtracted, not added again.'),
            ],
            scheme=[
                _scheme(1, 'x = 50', 2.0, 'Complete the algebra accurately and state the final result'),
            ],
            feedback=[
                _fb(1, 'You identified the relevant algebra topic and attempted a solution.',
                    missing='After clearing the denominator, the added constant must be subtracted, not added again. The correct result is x = 50.',
                    deduction='Full marks lost because the stated result does not follow the required algebraic rule.',
                    improve='Multiply both sides by the denominator, then use the inverse operation to isolate x.'),
            ],
            topic='equations_with_fractions', difficulty='difficult',
        ),
        _example(
            'Solve 2x + 3 > 7', 'linear inequality', 2.0, 2.0,
            steps=[
                _step(1, 'x > 2', 'correct', 2.0),
            ],
            scheme=[
                _scheme(1, 'x > 2', 2.0, 'Complete the algebra accurately and state the final result'),
            ],
            feedback=[
                _fb(1, 'You correctly isolated x and handled the inequality direction.',
                    improve='Isolate the variable and remember to reverse the sign only when multiplying or dividing by a negative value.'),
            ],
            topic='linear_inequalities', difficulty='easy',
        ),
        _example(
            'Solve -3x + 5 < -4', 'linear inequality', 1.0, 2.0,
            steps=[
                _step(1, '-3x < -9', 'partial', 1.0, 'The x term was isolated, but the final division and inequality direction were not completed.'),
            ],
            scheme=[
                _scheme(1, 'x > 3', 2.0, 'Complete the algebra accurately and state the final result'),
            ],
            feedback=[
                _fb(1, 'You made a valid start and used part of the correct algebraic method.',
                    missing='The x term was isolated, but the final division and inequality direction were not completed. The complete result is x > 3.',
                    deduction='1 mark deducted because the response is incomplete or not fully simplified.',
                    improve='Isolate the variable and remember to reverse the sign only when multiplying or dividing by a negative value.'),
            ],
            topic='linear_inequalities', difficulty='medium',
        ),
        _example(
            'Solve 4x + 5 > 21', 'linear inequality', 1.0, 2.0,
            steps=[
                _step(1, '4x > 16', 'partial', 1.0, 'The x term was isolated, but the final division and inequality direction were not completed.'),
            ],
            scheme=[
                _scheme(1, 'x > 4', 2.0, 'Complete the algebra accurately and state the final result'),
            ],
            feedback=[
                _fb(1, 'You made a valid start and used part of the correct algebraic method.',
                    missing='The x term was isolated, but the final division and inequality direction were not completed. The complete result is x > 4.',
                    deduction='1 mark deducted because the response is incomplete or not fully simplified.',
                    improve='Isolate the variable and remember to reverse the sign only when multiplying or dividing by a negative value.'),
            ],
            topic='linear_inequalities', difficulty='difficult',
        ),
        _example(
            'Solve -5x + 7 < -18', 'linear inequality', 0.0, 2.0,
            steps=[
                _step(1, 'x < 5', 'incorrect', 0.0, 'Dividing an inequality by a negative number reverses its direction.'),
            ],
            scheme=[
                _scheme(1, 'x > 5', 2.0, 'Complete the algebra accurately and state the final result'),
            ],
            feedback=[
                _fb(1, 'You identified the relevant algebra topic and attempted a solution.',
                    missing='Dividing an inequality by a negative number reverses its direction. The correct result is x > 5.',
                    deduction='Full marks lost because the stated result does not follow the required algebraic rule.',
                    improve='Isolate the variable and remember to reverse the sign only when multiplying or dividing by a negative value.'),
            ],
            topic='linear_inequalities', difficulty='medium',
        ),
        _example(
            'Solve 6x + 7 > 43', 'linear inequality', 0.0, 2.0,
            steps=[
                _step(1, 'x < 6', 'incorrect', 0.0, 'The inequality direction should remain unchanged when dividing by a positive number.'),
            ],
            scheme=[
                _scheme(1, 'x > 6', 2.0, 'Complete the algebra accurately and state the final result'),
            ],
            feedback=[
                _fb(1, 'You identified the relevant algebra topic and attempted a solution.',
                    missing='The inequality direction should remain unchanged when dividing by a positive number. The correct result is x > 6.',
                    deduction='Full marks lost because the stated result does not follow the required algebraic rule.',
                    improve='Isolate the variable and remember to reverse the sign only when multiplying or dividing by a negative value.'),
            ],
            topic='linear_inequalities', difficulty='difficult',
        ),
        _example(
            'Solve x^2 - 5x + 4 > 0', 'quadratic inequality', 2.0, 2.0,
            steps=[
                _step(1, 'x < 1 or x > 4', 'correct', 2.0),
            ],
            scheme=[
                _scheme(1, 'x < 1 or x > 4', 2.0, 'Complete the algebra accurately and state the final result'),
            ],
            feedback=[
                _fb(1, 'You correctly factorised the quadratic and selected the outside-root region.',
                    improve='Factorise, mark both roots on a sign diagram, and test the intervals before stating the solution set.'),
            ],
            topic='quadratic_inequalities', difficulty='easy',
        ),
        _example(
            'Solve x^2 - 7x + 10 < 0', 'quadratic inequality', 1.0, 2.0,
            steps=[
                _step(1, 'x < 5', 'partial', 1.0, 'Only one boundary or interval was stated, so the full solution set is incomplete.'),
            ],
            scheme=[
                _scheme(1, '2 < x < 5', 2.0, 'Complete the algebra accurately and state the final result'),
            ],
            feedback=[
                _fb(1, 'You made a valid start and used part of the correct algebraic method.',
                    missing='Only one boundary or interval was stated, so the full solution set is incomplete. The complete result is 2 < x < 5.',
                    deduction='1 mark deducted because the response is incomplete or not fully simplified.',
                    improve='Factorise, mark both roots on a sign diagram, and test the intervals before stating the solution set.'),
            ],
            topic='quadratic_inequalities', difficulty='medium',
        ),
        _example(
            'Solve x^2 - 9x + 18 > 0', 'quadratic inequality', 1.0, 2.0,
            steps=[
                _step(1, 'x > 6', 'partial', 1.0, 'Only one boundary or interval was stated, so the full solution set is incomplete.'),
            ],
            scheme=[
                _scheme(1, 'x < 3 or x > 6', 2.0, 'Complete the algebra accurately and state the final result'),
            ],
            feedback=[
                _fb(1, 'You made a valid start and used part of the correct algebraic method.',
                    missing='Only one boundary or interval was stated, so the full solution set is incomplete. The complete result is x < 3 or x > 6.',
                    deduction='1 mark deducted because the response is incomplete or not fully simplified.',
                    improve='Factorise, mark both roots on a sign diagram, and test the intervals before stating the solution set.'),
            ],
            topic='quadratic_inequalities', difficulty='difficult',
        ),
        _example(
            'Solve x^2 - 11x + 28 < 0', 'quadratic inequality', 0.0, 2.0,
            steps=[
                _step(1, 'x < 4 or x > 7', 'incorrect', 0.0, 'For this upward-opening quadratic, the required sign occurs between the two roots.'),
            ],
            scheme=[
                _scheme(1, '4 < x < 7', 2.0, 'Complete the algebra accurately and state the final result'),
            ],
            feedback=[
                _fb(1, 'You identified the relevant algebra topic and attempted a solution.',
                    missing='For this upward-opening quadratic, the required sign occurs between the two roots. The correct result is 4 < x < 7.',
                    deduction='Full marks lost because the stated result does not follow the required algebraic rule.',
                    improve='Factorise, mark both roots on a sign diagram, and test the intervals before stating the solution set.'),
            ],
            topic='quadratic_inequalities', difficulty='medium',
        ),
        _example(
            'Solve x^2 - 13x + 40 > 0', 'quadratic inequality', 0.0, 2.0,
            steps=[
                _step(1, '5 < x < 8', 'incorrect', 0.0, 'For this upward-opening quadratic, the required sign occurs outside the two roots.'),
            ],
            scheme=[
                _scheme(1, 'x < 5 or x > 8', 2.0, 'Complete the algebra accurately and state the final result'),
            ],
            feedback=[
                _fb(1, 'You identified the relevant algebra topic and attempted a solution.',
                    missing='For this upward-opening quadratic, the required sign occurs outside the two roots. The correct result is x < 5 or x > 8.',
                    deduction='Full marks lost because the stated result does not follow the required algebraic rule.',
                    improve='Factorise, mark both roots on a sign diagram, and test the intervals before stating the solution set.'),
            ],
            topic='quadratic_inequalities', difficulty='difficult',
        ),
        _example(
            'Solve y = 2x - 1, 3x + y = 9', 'substitution', 2.0, 2.0,
            steps=[
                _step(1, 'x = 2, y = 3', 'correct', 2.0),
            ],
            scheme=[
                _scheme(1, 'x = 2, y = 3', 2.0, 'Complete the algebra accurately and state the final result'),
            ],
            feedback=[
                _fb(1, 'You correctly substituted one equation into the other and found both variables.',
                    improve='After finding one variable, always substitute it into an original equation to calculate and verify the other.'),
            ],
            topic='simultaneous_substitution', difficulty='easy',
        ),
        _example(
            'Solve y = 3x - 4, 4x + y = 17', 'substitution', 1.0, 2.0,
            steps=[
                _step(1, 'x = 3', 'partial', 1.0, 'The value of x is correct, but y was not found by substituting back.'),
            ],
            scheme=[
                _scheme(1, 'x = 3, y = 5', 2.0, 'Complete the algebra accurately and state the final result'),
            ],
            feedback=[
                _fb(1, 'You made a valid start and used part of the correct algebraic method.',
                    missing='The value of x is correct, but y was not found by substituting back. The complete result is x = 3, y = 5.',
                    deduction='1 mark deducted because the response is incomplete or not fully simplified.',
                    improve='After finding one variable, always substitute it into an original equation to calculate and verify the other.'),
            ],
            topic='simultaneous_substitution', difficulty='medium',
        ),
        _example(
            'Solve y = 4x - 9, 5x + y = 27', 'substitution', 1.0, 2.0,
            steps=[
                _step(1, 'x = 4', 'partial', 1.0, 'The value of x is correct, but y was not found by substituting back.'),
            ],
            scheme=[
                _scheme(1, 'x = 4, y = 7', 2.0, 'Complete the algebra accurately and state the final result'),
            ],
            feedback=[
                _fb(1, 'You made a valid start and used part of the correct algebraic method.',
                    missing='The value of x is correct, but y was not found by substituting back. The complete result is x = 4, y = 7.',
                    deduction='1 mark deducted because the response is incomplete or not fully simplified.',
                    improve='After finding one variable, always substitute it into an original equation to calculate and verify the other.'),
            ],
            topic='simultaneous_substitution', difficulty='difficult',
        ),
        _example(
            'Solve y = 5x - 16, 6x + y = 39', 'substitution', 0.0, 2.0,
            steps=[
                _step(1, 'x = 5, y = 10', 'incorrect', 0.0, 'The x value was not substituted accurately into the first equation when calculating y.'),
            ],
            scheme=[
                _scheme(1, 'x = 5, y = 9', 2.0, 'Complete the algebra accurately and state the final result'),
            ],
            feedback=[
                _fb(1, 'You identified the relevant algebra topic and attempted a solution.',
                    missing='The x value was not substituted accurately into the first equation when calculating y. The correct result is x = 5, y = 9.',
                    deduction='Full marks lost because the stated result does not follow the required algebraic rule.',
                    improve='After finding one variable, always substitute it into an original equation to calculate and verify the other.'),
            ],
            topic='simultaneous_substitution', difficulty='medium',
        ),
        _example(
            'Solve y = 6x - 25, 7x + y = 53', 'substitution', 0.0, 2.0,
            steps=[
                _step(1, 'x = 6, y = 12', 'incorrect', 0.0, 'The x value was not substituted accurately into the first equation when calculating y.'),
            ],
            scheme=[
                _scheme(1, 'x = 6, y = 11', 2.0, 'Complete the algebra accurately and state the final result'),
            ],
            feedback=[
                _fb(1, 'You identified the relevant algebra topic and attempted a solution.',
                    missing='The x value was not substituted accurately into the first equation when calculating y. The correct result is x = 6, y = 11.',
                    deduction='Full marks lost because the stated result does not follow the required algebraic rule.',
                    improve='After finding one variable, always substitute it into an original equation to calculate and verify the other.'),
            ],
            topic='simultaneous_substitution', difficulty='difficult',
        ),
        _example(
            'Solve x + y = 5, x - y = 3', 'elimination', 2.0, 2.0,
            steps=[
                _step(1, 'x = 4, y = 1', 'correct', 2.0),
            ],
            scheme=[
                _scheme(1, 'x = 4, y = 1', 2.0, 'Complete the algebra accurately and state the final result'),
            ],
            feedback=[
                _fb(1, 'You correctly added the equations to eliminate y and then found both variables.',
                    improve='Use elimination for one variable, substitute back for the second, and check both equations.'),
            ],
            topic='simultaneous_elimination', difficulty='easy',
        ),
        _example(
            'Solve x + y = 7, x - y = 3', 'elimination', 1.0, 2.0,
            steps=[
                _step(1, '2x = 10, so x = 5', 'partial', 1.0, 'Elimination correctly produced x, but y was not calculated by substitution.'),
            ],
            scheme=[
                _scheme(1, 'x = 5, y = 2', 2.0, 'Complete the algebra accurately and state the final result'),
            ],
            feedback=[
                _fb(1, 'You made a valid start and used part of the correct algebraic method.',
                    missing='Elimination correctly produced x, but y was not calculated by substitution. The complete result is x = 5, y = 2.',
                    deduction='1 mark deducted because the response is incomplete or not fully simplified.',
                    improve='Use elimination for one variable, substitute back for the second, and check both equations.'),
            ],
            topic='simultaneous_elimination', difficulty='medium',
        ),
        _example(
            'Solve x + y = 9, x - y = 3', 'elimination', 1.0, 2.0,
            steps=[
                _step(1, '2x = 12, so x = 6', 'partial', 1.0, 'Elimination correctly produced x, but y was not calculated by substitution.'),
            ],
            scheme=[
                _scheme(1, 'x = 6, y = 3', 2.0, 'Complete the algebra accurately and state the final result'),
            ],
            feedback=[
                _fb(1, 'You made a valid start and used part of the correct algebraic method.',
                    missing='Elimination correctly produced x, but y was not calculated by substitution. The complete result is x = 6, y = 3.',
                    deduction='1 mark deducted because the response is incomplete or not fully simplified.',
                    improve='Use elimination for one variable, substitute back for the second, and check both equations.'),
            ],
            topic='simultaneous_elimination', difficulty='difficult',
        ),
        _example(
            'Solve x + y = 11, x - y = 3', 'elimination', 0.0, 2.0,
            steps=[
                _step(1, 'x = 4, y = 7', 'incorrect', 0.0, 'The two variable values were interchanged and do not satisfy both original equations.'),
            ],
            scheme=[
                _scheme(1, 'x = 7, y = 4', 2.0, 'Complete the algebra accurately and state the final result'),
            ],
            feedback=[
                _fb(1, 'You identified the relevant algebra topic and attempted a solution.',
                    missing='The two variable values were interchanged and do not satisfy both original equations. The correct result is x = 7, y = 4.',
                    deduction='Full marks lost because the stated result does not follow the required algebraic rule.',
                    improve='Use elimination for one variable, substitute back for the second, and check both equations.'),
            ],
            topic='simultaneous_elimination', difficulty='medium',
        ),
        _example(
            'Solve x + y = 13, x - y = 3', 'elimination', 0.0, 2.0,
            steps=[
                _step(1, 'x = 5, y = 8', 'incorrect', 0.0, 'The two variable values were interchanged and do not satisfy both original equations.'),
            ],
            scheme=[
                _scheme(1, 'x = 8, y = 5', 2.0, 'Complete the algebra accurately and state the final result'),
            ],
            feedback=[
                _fb(1, 'You identified the relevant algebra topic and attempted a solution.',
                    missing='The two variable values were interchanged and do not satisfy both original equations. The correct result is x = 8, y = 5.',
                    deduction='Full marks lost because the stated result does not follow the required algebraic rule.',
                    improve='Use elimination for one variable, substitute back for the second, and check both equations.'),
            ],
            topic='simultaneous_elimination', difficulty='difficult',
        ),
        _example(
            'Solve y = x + 3, y = x^2 + 1', 'substitution (linear-quadratic)', 2.0, 2.0,
            steps=[
                _step(1, '(2, 5) and (-1, 2)', 'correct', 2.0),
            ],
            scheme=[
                _scheme(1, '(2, 5) and (-1, 2)', 2.0, 'Complete the algebra accurately and state the final result'),
            ],
            feedback=[
                _fb(1, 'You correctly substituted, solved the quadratic, and matched each x value with its y value.',
                    improve='A quadratic normally gives two roots; substitute each root into the linear equation to obtain both points.'),
            ],
            topic='simultaneous_linear_quadratic', difficulty='easy',
        ),
        _example(
            'Solve y = x + 4, y = x^2 - 2', 'substitution (linear-quadratic)', 1.0, 2.0,
            steps=[
                _step(1, '(3, 7)', 'partial', 1.0, 'Only one of the two intersection points was stated.'),
            ],
            scheme=[
                _scheme(1, '(3, 7) and (-2, 2)', 2.0, 'Complete the algebra accurately and state the final result'),
            ],
            feedback=[
                _fb(1, 'You made a valid start and used part of the correct algebraic method.',
                    missing='Only one of the two intersection points was stated. The complete result is (3, 7) and (-2, 2).',
                    deduction='1 mark deducted because the response is incomplete or not fully simplified.',
                    improve='A quadratic normally gives two roots; substitute each root into the linear equation to obtain both points.'),
            ],
            topic='simultaneous_linear_quadratic', difficulty='medium',
        ),
        _example(
            'Solve y = x + 5, y = x^2 - 7', 'substitution (linear-quadratic)', 1.0, 2.0,
            steps=[
                _step(1, '(4, 9)', 'partial', 1.0, 'Only one of the two intersection points was stated.'),
            ],
            scheme=[
                _scheme(1, '(4, 9) and (-3, 2)', 2.0, 'Complete the algebra accurately and state the final result'),
            ],
            feedback=[
                _fb(1, 'You made a valid start and used part of the correct algebraic method.',
                    missing='Only one of the two intersection points was stated. The complete result is (4, 9) and (-3, 2).',
                    deduction='1 mark deducted because the response is incomplete or not fully simplified.',
                    improve='A quadratic normally gives two roots; substitute each root into the linear equation to obtain both points.'),
            ],
            topic='simultaneous_linear_quadratic', difficulty='difficult',
        ),
        _example(
            'Solve y = x + 6, y = x^2 - 14', 'substitution (linear-quadratic)', 0.0, 2.0,
            steps=[
                _step(1, '(5, 11) and (4, 10)', 'incorrect', 0.0, 'The sign of the second quadratic root was changed, producing a point that is not on both curves.'),
            ],
            scheme=[
                _scheme(1, '(5, 11) and (-4, 2)', 2.0, 'Complete the algebra accurately and state the final result'),
            ],
            feedback=[
                _fb(1, 'You identified the relevant algebra topic and attempted a solution.',
                    missing='The sign of the second quadratic root was changed, producing a point that is not on both curves. The correct result is (5, 11) and (-4, 2).',
                    deduction='Full marks lost because the stated result does not follow the required algebraic rule.',
                    improve='A quadratic normally gives two roots; substitute each root into the linear equation to obtain both points.'),
            ],
            topic='simultaneous_linear_quadratic', difficulty='medium',
        ),
        _example(
            'Solve y = x + 7, y = x^2 - 23', 'substitution (linear-quadratic)', 0.0, 2.0,
            steps=[
                _step(1, '(6, 13) and (5, 12)', 'incorrect', 0.0, 'The sign of the second quadratic root was changed, producing a point that is not on both curves.'),
            ],
            scheme=[
                _scheme(1, '(6, 13) and (-5, 2)', 2.0, 'Complete the algebra accurately and state the final result'),
            ],
            feedback=[
                _fb(1, 'You identified the relevant algebra topic and attempted a solution.',
                    missing='The sign of the second quadratic root was changed, producing a point that is not on both curves. The correct result is (6, 13) and (-5, 2).',
                    deduction='Full marks lost because the stated result does not follow the required algebraic rule.',
                    improve='A quadratic normally gives two roots; substitute each root into the linear equation to obtain both points.'),
            ],
            topic='simultaneous_linear_quadratic', difficulty='difficult',
        ),
        _example(
            'Solve x^2 - 7x + 10 = 0 by factorisation', 'quadratic factorisation', 2.0, 2.0,
            steps=[
                _step(1, 'x = 2 or x = 5', 'correct', 2.0),
            ],
            scheme=[
                _scheme(1, 'x = 2 or x = 5', 2.0, 'Complete the algebra accurately and state the final result'),
            ],
            feedback=[
                _fb(1, 'You correctly factorised the quadratic and applied the zero-product property.',
                    improve='After factorising, set each factor equal to zero and solve each resulting linear equation.'),
            ],
            topic='quadratic_factorisation', difficulty='easy',
        ),
        _example(
            'Solve x^2 - 9x + 18 = 0 by factorisation', 'quadratic factorisation', 1.0, 2.0,
            steps=[
                _step(1, '(x - 3)(x - 6) = 0', 'partial', 1.0, 'The factorisation is correct, but the two roots were not stated.'),
            ],
            scheme=[
                _scheme(1, 'x = 3 or x = 6', 2.0, 'Complete the algebra accurately and state the final result'),
            ],
            feedback=[
                _fb(1, 'You made a valid start and used part of the correct algebraic method.',
                    missing='The factorisation is correct, but the two roots were not stated. The complete result is x = 3 or x = 6.',
                    deduction='1 mark deducted because the response is incomplete or not fully simplified.',
                    improve='After factorising, set each factor equal to zero and solve each resulting linear equation.'),
            ],
            topic='quadratic_factorisation', difficulty='medium',
        ),
        _example(
            'Solve x^2 - 11x + 28 = 0 by factorisation', 'quadratic factorisation', 1.0, 2.0,
            steps=[
                _step(1, '(x - 4)(x - 7) = 0', 'partial', 1.0, 'The factorisation is correct, but the two roots were not stated.'),
            ],
            scheme=[
                _scheme(1, 'x = 4 or x = 7', 2.0, 'Complete the algebra accurately and state the final result'),
            ],
            feedback=[
                _fb(1, 'You made a valid start and used part of the correct algebraic method.',
                    missing='The factorisation is correct, but the two roots were not stated. The complete result is x = 4 or x = 7.',
                    deduction='1 mark deducted because the response is incomplete or not fully simplified.',
                    improve='After factorising, set each factor equal to zero and solve each resulting linear equation.'),
            ],
            topic='quadratic_factorisation', difficulty='difficult',
        ),
        _example(
            'Solve x^2 - 13x + 40 = 0 by factorisation', 'quadratic factorisation', 0.0, 2.0,
            steps=[
                _step(1, 'x = -5 or x = -8', 'incorrect', 0.0, 'From (x-a)=0 the root is x=a, so the signs of both roots were reversed.'),
            ],
            scheme=[
                _scheme(1, 'x = 5 or x = 8', 2.0, 'Complete the algebra accurately and state the final result'),
            ],
            feedback=[
                _fb(1, 'You identified the relevant algebra topic and attempted a solution.',
                    missing='From (x-a)=0 the root is x=a, so the signs of both roots were reversed. The correct result is x = 5 or x = 8.',
                    deduction='Full marks lost because the stated result does not follow the required algebraic rule.',
                    improve='After factorising, set each factor equal to zero and solve each resulting linear equation.'),
            ],
            topic='quadratic_factorisation', difficulty='medium',
        ),
        _example(
            'Solve x^2 - 15x + 54 = 0 by factorisation', 'quadratic factorisation', 0.0, 2.0,
            steps=[
                _step(1, 'x = -6 or x = -9', 'incorrect', 0.0, 'From (x-a)=0 the root is x=a, so the signs of both roots were reversed.'),
            ],
            scheme=[
                _scheme(1, 'x = 6 or x = 9', 2.0, 'Complete the algebra accurately and state the final result'),
            ],
            feedback=[
                _fb(1, 'You identified the relevant algebra topic and attempted a solution.',
                    missing='From (x-a)=0 the root is x=a, so the signs of both roots were reversed. The correct result is x = 6 or x = 9.',
                    deduction='Full marks lost because the stated result does not follow the required algebraic rule.',
                    improve='After factorising, set each factor equal to zero and solve each resulting linear equation.'),
            ],
            topic='quadratic_factorisation', difficulty='difficult',
        ),
        _example(
            'Solve x^2 + 2x - 3 = 0 using the quadratic formula', 'quadratic formula', 2.0, 2.0,
            steps=[
                _step(1, 'x = 1 or x = -3', 'correct', 2.0),
            ],
            scheme=[
                _scheme(1, 'x = 1 or x = -3', 2.0, 'Complete the algebra accurately and state the final result'),
            ],
            feedback=[
                _fb(1, 'You correctly substituted a, b, and c into the quadratic formula and evaluated both roots.',
                    improve='Write a, b, and c with signs, substitute into -b±√(b²-4ac), and evaluate both cases separately.'),
            ],
            topic='quadratic_formula', difficulty='easy',
        ),
        _example(
            'Solve x^2 + 2x - 8 = 0 using the quadratic formula', 'quadratic formula', 1.0, 2.0,
            steps=[
                _step(1, 'x = (-2 ± √(36))/2', 'partial', 1.0, 'The formula was set up correctly, but the square root and the two final cases were not evaluated.'),
            ],
            scheme=[
                _scheme(1, 'x = 2 or x = -4', 2.0, 'Complete the algebra accurately and state the final result'),
            ],
            feedback=[
                _fb(1, 'You made a valid start and used part of the correct algebraic method.',
                    missing='The formula was set up correctly, but the square root and the two final cases were not evaluated. The complete result is x = 2 or x = -4.',
                    deduction='1 mark deducted because the response is incomplete or not fully simplified.',
                    improve='Write a, b, and c with signs, substitute into -b±√(b²-4ac), and evaluate both cases separately.'),
            ],
            topic='quadratic_formula', difficulty='medium',
        ),
        _example(
            'Solve x^2 + 2x - 15 = 0 using the quadratic formula', 'quadratic formula', 1.0, 2.0,
            steps=[
                _step(1, 'x = (-2 ± √(64))/2', 'partial', 1.0, 'The formula was set up correctly, but the square root and the two final cases were not evaluated.'),
            ],
            scheme=[
                _scheme(1, 'x = 3 or x = -5', 2.0, 'Complete the algebra accurately and state the final result'),
            ],
            feedback=[
                _fb(1, 'You made a valid start and used part of the correct algebraic method.',
                    missing='The formula was set up correctly, but the square root and the two final cases were not evaluated. The complete result is x = 3 or x = -5.',
                    deduction='1 mark deducted because the response is incomplete or not fully simplified.',
                    improve='Write a, b, and c with signs, substitute into -b±√(b²-4ac), and evaluate both cases separately.'),
            ],
            topic='quadratic_formula', difficulty='difficult',
        ),
        _example(
            'Solve x^2 + 2x - 24 = 0 using the quadratic formula', 'quadratic formula', 0.0, 2.0,
            steps=[
                _step(1, 'x = -4 or x = 6', 'incorrect', 0.0, 'A sign was mishandled when using -b or evaluating the ± cases, reversing the roots.'),
            ],
            scheme=[
                _scheme(1, 'x = 4 or x = -6', 2.0, 'Complete the algebra accurately and state the final result'),
            ],
            feedback=[
                _fb(1, 'You identified the relevant algebra topic and attempted a solution.',
                    missing='A sign was mishandled when using -b or evaluating the ± cases, reversing the roots. The correct result is x = 4 or x = -6.',
                    deduction='Full marks lost because the stated result does not follow the required algebraic rule.',
                    improve='Write a, b, and c with signs, substitute into -b±√(b²-4ac), and evaluate both cases separately.'),
            ],
            topic='quadratic_formula', difficulty='medium',
        ),
        _example(
            'Solve x^2 + 2x - 35 = 0 using the quadratic formula', 'quadratic formula', 0.0, 2.0,
            steps=[
                _step(1, 'x = -5 or x = 7', 'incorrect', 0.0, 'A sign was mishandled when using -b or evaluating the ± cases, reversing the roots.'),
            ],
            scheme=[
                _scheme(1, 'x = 5 or x = -7', 2.0, 'Complete the algebra accurately and state the final result'),
            ],
            feedback=[
                _fb(1, 'You identified the relevant algebra topic and attempted a solution.',
                    missing='A sign was mishandled when using -b or evaluating the ± cases, reversing the roots. The correct result is x = 5 or x = -7.',
                    deduction='Full marks lost because the stated result does not follow the required algebraic rule.',
                    improve='Write a, b, and c with signs, substitute into -b±√(b²-4ac), and evaluate both cases separately.'),
            ],
            topic='quadratic_formula', difficulty='difficult',
        ),
        _example(
            'Solve x^2 + 10x + 21 = 0 by completing the square', 'completing the square', 2.0, 2.0,
            steps=[
                _step(1, 'x = -3 or x = -7', 'correct', 2.0),
            ],
            scheme=[
                _scheme(1, 'x = -3 or x = -7', 2.0, 'Complete the algebra accurately and state the final result'),
            ],
            feedback=[
                _fb(1, 'You correctly completed the square and included both square-root cases.',
                    improve='Use half the x coefficient inside the bracket, balance the added square, then use ± when taking square roots.'),
            ],
            topic='completing_the_square', difficulty='easy',
        ),
        _example(
            'Solve x^2 + 12x + 27 = 0 by completing the square', 'completing the square', 1.0, 2.0,
            steps=[
                _step(1, '(x + 6)^2 = 9, so x = -3', 'partial', 1.0, 'The completed-square equation is correct, but only the positive square-root case was used.'),
            ],
            scheme=[
                _scheme(1, 'x = -3 or x = -9', 2.0, 'Complete the algebra accurately and state the final result'),
            ],
            feedback=[
                _fb(1, 'You made a valid start and used part of the correct algebraic method.',
                    missing='The completed-square equation is correct, but only the positive square-root case was used. The complete result is x = -3 or x = -9.',
                    deduction='1 mark deducted because the response is incomplete or not fully simplified.',
                    improve='Use half the x coefficient inside the bracket, balance the added square, then use ± when taking square roots.'),
            ],
            topic='completing_the_square', difficulty='medium',
        ),
        _example(
            'Solve x^2 + 14x + 33 = 0 by completing the square', 'completing the square', 1.0, 2.0,
            steps=[
                _step(1, '(x + 7)^2 = 16, so x = -3', 'partial', 1.0, 'The completed-square equation is correct, but only the positive square-root case was used.'),
            ],
            scheme=[
                _scheme(1, 'x = -3 or x = -11', 2.0, 'Complete the algebra accurately and state the final result'),
            ],
            feedback=[
                _fb(1, 'You made a valid start and used part of the correct algebraic method.',
                    missing='The completed-square equation is correct, but only the positive square-root case was used. The complete result is x = -3 or x = -11.',
                    deduction='1 mark deducted because the response is incomplete or not fully simplified.',
                    improve='Use half the x coefficient inside the bracket, balance the added square, then use ± when taking square roots.'),
            ],
            topic='completing_the_square', difficulty='difficult',
        ),
        _example(
            'Solve x^2 + 16x + 39 = 0 by completing the square', 'completing the square', 0.0, 2.0,
            steps=[
                _step(1, '(x + 16)^2 = 25', 'incorrect', 0.0, 'The number inside the completed square must be half the x coefficient, not the full coefficient.'),
            ],
            scheme=[
                _scheme(1, 'x = -3 or x = -13', 2.0, 'Complete the algebra accurately and state the final result'),
            ],
            feedback=[
                _fb(1, 'You identified the relevant algebra topic and attempted a solution.',
                    missing='The number inside the completed square must be half the x coefficient, not the full coefficient. The correct result is x = -3 or x = -13.',
                    deduction='Full marks lost because the stated result does not follow the required algebraic rule.',
                    improve='Use half the x coefficient inside the bracket, balance the added square, then use ± when taking square roots.'),
            ],
            topic='completing_the_square', difficulty='medium',
        ),
        _example(
            'Solve x^2 + 18x + 45 = 0 by completing the square', 'completing the square', 0.0, 2.0,
            steps=[
                _step(1, '(x + 18)^2 = 36', 'incorrect', 0.0, 'The number inside the completed square must be half the x coefficient, not the full coefficient.'),
            ],
            scheme=[
                _scheme(1, 'x = -3 or x = -15', 2.0, 'Complete the algebra accurately and state the final result'),
            ],
            feedback=[
                _fb(1, 'You identified the relevant algebra topic and attempted a solution.',
                    missing='The number inside the completed square must be half the x coefficient, not the full coefficient. The correct result is x = -3 or x = -15.',
                    deduction='Full marks lost because the stated result does not follow the required algebraic rule.',
                    improve='Use half the x coefficient inside the bracket, balance the added square, then use ± when taking square roots.'),
            ],
            topic='completing_the_square', difficulty='difficult',
        ),
        _example(
            'Simplify (x^2 - 9)/(x - 3)', 'simplifying algebraic fractions', 2.0, 2.0,
            steps=[
                _step(1, 'x + 3, x ≠ 3', 'correct', 2.0),
            ],
            scheme=[
                _scheme(1, 'x + 3, x ≠ 3', 2.0, 'Complete the algebra accurately and state the final result'),
            ],
            feedback=[
                _fb(1, 'You correctly factorised the numerator, cancelled the common factor, and stated the restriction.',
                    improve='Factorise first, cancel only common factors, and retain restrictions from the original denominator.'),
            ],
            topic='algebraic_fractions', difficulty='easy',
        ),
        _example(
            'Simplify (x^2 - 16)/(x - 4)', 'simplifying algebraic fractions', 1.0, 2.0,
            steps=[
                _step(1, 'x + 4', 'partial', 1.0, 'The simplification is correct, but the excluded value from the original denominator was not stated.'),
            ],
            scheme=[
                _scheme(1, 'x + 4, x ≠ 4', 2.0, 'Complete the algebra accurately and state the final result'),
            ],
            feedback=[
                _fb(1, 'You made a valid start and used part of the correct algebraic method.',
                    missing='The simplification is correct, but the excluded value from the original denominator was not stated. The complete result is x + 4, x ≠ 4.',
                    deduction='1 mark deducted because the response is incomplete or not fully simplified.',
                    improve='Factorise first, cancel only common factors, and retain restrictions from the original denominator.'),
            ],
            topic='algebraic_fractions', difficulty='medium',
        ),
        _example(
            'Simplify (x^2 - 25)/(x - 5)', 'simplifying algebraic fractions', 1.0, 2.0,
            steps=[
                _step(1, 'x + 5', 'partial', 1.0, 'The simplification is correct, but the excluded value from the original denominator was not stated.'),
            ],
            scheme=[
                _scheme(1, 'x + 5, x ≠ 5', 2.0, 'Complete the algebra accurately and state the final result'),
            ],
            feedback=[
                _fb(1, 'You made a valid start and used part of the correct algebraic method.',
                    missing='The simplification is correct, but the excluded value from the original denominator was not stated. The complete result is x + 5, x ≠ 5.',
                    deduction='1 mark deducted because the response is incomplete or not fully simplified.',
                    improve='Factorise first, cancel only common factors, and retain restrictions from the original denominator.'),
            ],
            topic='algebraic_fractions', difficulty='difficult',
        ),
        _example(
            'Simplify (x^2 - 36)/(x - 6)', 'simplifying algebraic fractions', 0.0, 2.0,
            steps=[
                _step(1, 'x - 6', 'incorrect', 0.0, 'After factorising (x-a)(x+a), cancelling (x-a) leaves x+a, not x-a.'),
            ],
            scheme=[
                _scheme(1, 'x + 6, x ≠ 6', 2.0, 'Complete the algebra accurately and state the final result'),
            ],
            feedback=[
                _fb(1, 'You identified the relevant algebra topic and attempted a solution.',
                    missing='After factorising (x-a)(x+a), cancelling (x-a) leaves x+a, not x-a. The correct result is x + 6, x ≠ 6.',
                    deduction='Full marks lost because the stated result does not follow the required algebraic rule.',
                    improve='Factorise first, cancel only common factors, and retain restrictions from the original denominator.'),
            ],
            topic='algebraic_fractions', difficulty='medium',
        ),
        _example(
            'Simplify (x^2 - 49)/(x - 7)', 'simplifying algebraic fractions', 0.0, 2.0,
            steps=[
                _step(1, 'x - 7', 'incorrect', 0.0, 'After factorising (x-a)(x+a), cancelling (x-a) leaves x+a, not x-a.'),
            ],
            scheme=[
                _scheme(1, 'x + 7, x ≠ 7', 2.0, 'Complete the algebra accurately and state the final result'),
            ],
            feedback=[
                _fb(1, 'You identified the relevant algebra topic and attempted a solution.',
                    missing='After factorising (x-a)(x+a), cancelling (x-a) leaves x+a, not x-a. The correct result is x + 7, x ≠ 7.',
                    deduction='Full marks lost because the stated result does not follow the required algebraic rule.',
                    improve='Factorise first, cancel only common factors, and retain restrictions from the original denominator.'),
            ],
            topic='algebraic_fractions', difficulty='difficult',
        ),
        _example(
            'Add (2x^2 + 3x - 4) and (1x^2 - 5x + 2)', 'polynomial addition', 2.0, 2.0,
            steps=[
                _step(1, '3x^2 - 2x - 2', 'correct', 2.0),
            ],
            scheme=[
                _scheme(1, '3x^2 - 2x - 2', 2.0, 'Complete the algebra accurately and state the final result'),
            ],
            feedback=[
                _fb(1, 'You correctly combined coefficients of matching powers of x.',
                    improve='Align equal powers vertically, keep every sign, and then add their coefficients.'),
            ],
            topic='polynomial_addition_subtraction', difficulty='easy',
        ),
        _example(
            'Add (3x^2 + 5x - 5) and (2x^2 - 6x + 4)', 'polynomial addition', 1.0, 2.0,
            steps=[
                _step(1, '(5)x^2 + (5-6)x + (4-5)', 'partial', 1.0, 'The corresponding terms were paired correctly, but the arithmetic was left unevaluated.'),
            ],
            scheme=[
                _scheme(1, '5x^2 - 1x - 1', 2.0, 'Complete the algebra accurately and state the final result'),
            ],
            feedback=[
                _fb(1, 'You made a valid start and used part of the correct algebraic method.',
                    missing='The corresponding terms were paired correctly, but the arithmetic was left unevaluated. The complete result is 5x^2 - 1x - 1.',
                    deduction='1 mark deducted because the response is incomplete or not fully simplified.',
                    improve='Align equal powers vertically, keep every sign, and then add their coefficients.'),
            ],
            topic='polynomial_addition_subtraction', difficulty='medium',
        ),
        _example(
            'Add (4x^2 + 7x - 6) and (3x^2 - 7x + 6)', 'polynomial addition', 1.0, 2.0,
            steps=[
                _step(1, '(7)x^2 + (7-7)x + (6-6)', 'partial', 1.0, 'The corresponding terms were paired correctly, but the arithmetic was left unevaluated.'),
            ],
            scheme=[
                _scheme(1, '7x^2', 2.0, 'Complete the algebra accurately and state the final result'),
            ],
            feedback=[
                _fb(1, 'You made a valid start and used part of the correct algebraic method.',
                    missing='The corresponding terms were paired correctly, but the arithmetic was left unevaluated. The complete result is 7x^2.',
                    deduction='1 mark deducted because the response is incomplete or not fully simplified.',
                    improve='Align equal powers vertically, keep every sign, and then add their coefficients.'),
            ],
            topic='polynomial_addition_subtraction', difficulty='difficult',
        ),
        _example(
            'Add (5x^2 + 9x - 7) and (4x^2 - 8x + 8)', 'polynomial addition', 0.0, 2.0,
            steps=[
                _step(1, '9x^2 + 17x + 15', 'incorrect', 0.0, 'The negative signs in the second polynomial were ignored when combining the x and constant terms.'),
            ],
            scheme=[
                _scheme(1, '9x^2 + 1x + 1', 2.0, 'Complete the algebra accurately and state the final result'),
            ],
            feedback=[
                _fb(1, 'You identified the relevant algebra topic and attempted a solution.',
                    missing='The negative signs in the second polynomial were ignored when combining the x and constant terms. The correct result is 9x^2 + 1x + 1.',
                    deduction='Full marks lost because the stated result does not follow the required algebraic rule.',
                    improve='Align equal powers vertically, keep every sign, and then add their coefficients.'),
            ],
            topic='polynomial_addition_subtraction', difficulty='medium',
        ),
        _example(
            'Add (6x^2 + 11x - 8) and (5x^2 - 9x + 10)', 'polynomial addition', 0.0, 2.0,
            steps=[
                _step(1, '11x^2 + 20x + 18', 'incorrect', 0.0, 'The negative signs in the second polynomial were ignored when combining the x and constant terms.'),
            ],
            scheme=[
                _scheme(1, '11x^2 + 2x + 2', 2.0, 'Complete the algebra accurately and state the final result'),
            ],
            feedback=[
                _fb(1, 'You identified the relevant algebra topic and attempted a solution.',
                    missing='The negative signs in the second polynomial were ignored when combining the x and constant terms. The correct result is 11x^2 + 2x + 2.',
                    deduction='Full marks lost because the stated result does not follow the required algebraic rule.',
                    improve='Align equal powers vertically, keep every sign, and then add their coefficients.'),
            ],
            topic='polynomial_addition_subtraction', difficulty='difficult',
        ),
        _example(
            'Multiply (x + 2)(x^2 + 1x + 3)', 'polynomial multiplication', 2.0, 2.0,
            steps=[
                _step(1, 'x^3 + 3x^2 + 5x + 6', 'correct', 2.0),
            ],
            scheme=[
                _scheme(1, 'x^3 + 3x^2 + 5x + 6', 2.0, 'Complete the algebra accurately and state the final result'),
            ],
            feedback=[
                _fb(1, 'You correctly distributed both terms and collected matching powers.',
                    improve='Distribute each term across the full polynomial, then collect x³, x², x, and constants separately.'),
            ],
            topic='polynomial_multiplication', difficulty='easy',
        ),
        _example(
            'Multiply (x + 3)(x^2 + 2x + 4)', 'polynomial multiplication', 1.0, 2.0,
            steps=[
                _step(1, 'x^3 + 2x^2 + 4x + 3x^2 + 6x + 12', 'partial', 1.0, 'All products are present, but like powers of x were not combined.'),
            ],
            scheme=[
                _scheme(1, 'x^3 + 5x^2 + 10x + 12', 2.0, 'Complete the algebra accurately and state the final result'),
            ],
            feedback=[
                _fb(1, 'You made a valid start and used part of the correct algebraic method.',
                    missing='All products are present, but like powers of x were not combined. The complete result is x^3 + 5x^2 + 10x + 12.',
                    deduction='1 mark deducted because the response is incomplete or not fully simplified.',
                    improve='Distribute each term across the full polynomial, then collect x³, x², x, and constants separately.'),
            ],
            topic='polynomial_multiplication', difficulty='medium',
        ),
        _example(
            'Multiply (x + 4)(x^2 + 3x + 5)', 'polynomial multiplication', 1.0, 2.0,
            steps=[
                _step(1, 'x^3 + 3x^2 + 5x + 4x^2 + 12x + 20', 'partial', 1.0, 'All products are present, but like powers of x were not combined.'),
            ],
            scheme=[
                _scheme(1, 'x^3 + 7x^2 + 17x + 20', 2.0, 'Complete the algebra accurately and state the final result'),
            ],
            feedback=[
                _fb(1, 'You made a valid start and used part of the correct algebraic method.',
                    missing='All products are present, but like powers of x were not combined. The complete result is x^3 + 7x^2 + 17x + 20.',
                    deduction='1 mark deducted because the response is incomplete or not fully simplified.',
                    improve='Distribute each term across the full polynomial, then collect x³, x², x, and constants separately.'),
            ],
            topic='polynomial_multiplication', difficulty='difficult',
        ),
        _example(
            'Multiply (x + 5)(x^2 + 4x + 6)', 'polynomial multiplication', 0.0, 2.0,
            steps=[
                _step(1, 'x^3 + 9x^2 + 30', 'incorrect', 0.0, 'The x terms created by cross-products were omitted from the final polynomial.'),
            ],
            scheme=[
                _scheme(1, 'x^3 + 9x^2 + 26x + 30', 2.0, 'Complete the algebra accurately and state the final result'),
            ],
            feedback=[
                _fb(1, 'You identified the relevant algebra topic and attempted a solution.',
                    missing='The x terms created by cross-products were omitted from the final polynomial. The correct result is x^3 + 9x^2 + 26x + 30.',
                    deduction='Full marks lost because the stated result does not follow the required algebraic rule.',
                    improve='Distribute each term across the full polynomial, then collect x³, x², x, and constants separately.'),
            ],
            topic='polynomial_multiplication', difficulty='medium',
        ),
        _example(
            'Multiply (x + 6)(x^2 + 5x + 7)', 'polynomial multiplication', 0.0, 2.0,
            steps=[
                _step(1, 'x^3 + 11x^2 + 42', 'incorrect', 0.0, 'The x terms created by cross-products were omitted from the final polynomial.'),
            ],
            scheme=[
                _scheme(1, 'x^3 + 11x^2 + 37x + 42', 2.0, 'Complete the algebra accurately and state the final result'),
            ],
            feedback=[
                _fb(1, 'You identified the relevant algebra topic and attempted a solution.',
                    missing='The x terms created by cross-products were omitted from the final polynomial. The correct result is x^3 + 11x^2 + 37x + 42.',
                    deduction='Full marks lost because the stated result does not follow the required algebraic rule.',
                    improve='Distribute each term across the full polynomial, then collect x³, x², x, and constants separately.'),
            ],
            topic='polynomial_multiplication', difficulty='difficult',
        ),
        _example(
            'Divide x^2 + 6x + 8 by (x + 2)', 'polynomial division by factorisation', 2.0, 2.0,
            steps=[
                _step(1, 'x + 4', 'correct', 2.0),
            ],
            scheme=[
                _scheme(1, 'x + 4', 2.0, 'Complete the algebra accurately and state the final result'),
            ],
            feedback=[
                _fb(1, 'You correctly factorised the dividend and identified the remaining factor as the quotient.',
                    improve='Once the dividend is written as divisor × factor, cancel the divisor and state the remaining factor.'),
            ],
            topic='polynomial_division', difficulty='easy',
        ),
        _example(
            'Divide x^2 + 8x + 15 by (x + 3)', 'polynomial division by factorisation', 1.0, 2.0,
            steps=[
                _step(1, 'x^2 + 8x + 15 = (x + 3)(x + 5)', 'partial', 1.0, 'The dividend was factorised correctly, but the quotient was not explicitly stated.'),
            ],
            scheme=[
                _scheme(1, 'x + 5', 2.0, 'Complete the algebra accurately and state the final result'),
            ],
            feedback=[
                _fb(1, 'You made a valid start and used part of the correct algebraic method.',
                    missing='The dividend was factorised correctly, but the quotient was not explicitly stated. The complete result is x + 5.',
                    deduction='1 mark deducted because the response is incomplete or not fully simplified.',
                    improve='Once the dividend is written as divisor × factor, cancel the divisor and state the remaining factor.'),
            ],
            topic='polynomial_division', difficulty='medium',
        ),
        _example(
            'Divide x^2 + 10x + 24 by (x + 4)', 'polynomial division by factorisation', 1.0, 2.0,
            steps=[
                _step(1, 'x^2 + 10x + 24 = (x + 4)(x + 6)', 'partial', 1.0, 'The dividend was factorised correctly, but the quotient was not explicitly stated.'),
            ],
            scheme=[
                _scheme(1, 'x + 6', 2.0, 'Complete the algebra accurately and state the final result'),
            ],
            feedback=[
                _fb(1, 'You made a valid start and used part of the correct algebraic method.',
                    missing='The dividend was factorised correctly, but the quotient was not explicitly stated. The complete result is x + 6.',
                    deduction='1 mark deducted because the response is incomplete or not fully simplified.',
                    improve='Once the dividend is written as divisor × factor, cancel the divisor and state the remaining factor.'),
            ],
            topic='polynomial_division', difficulty='difficult',
        ),
        _example(
            'Divide x^2 + 12x + 35 by (x + 5)', 'polynomial division by factorisation', 0.0, 2.0,
            steps=[
                _step(1, 'x + 5', 'incorrect', 0.0, 'The divisor itself was repeated as the quotient instead of taking the other factor.'),
            ],
            scheme=[
                _scheme(1, 'x + 7', 2.0, 'Complete the algebra accurately and state the final result'),
            ],
            feedback=[
                _fb(1, 'You identified the relevant algebra topic and attempted a solution.',
                    missing='The divisor itself was repeated as the quotient instead of taking the other factor. The correct result is x + 7.',
                    deduction='Full marks lost because the stated result does not follow the required algebraic rule.',
                    improve='Once the dividend is written as divisor × factor, cancel the divisor and state the remaining factor.'),
            ],
            topic='polynomial_division', difficulty='medium',
        ),
        _example(
            'Divide x^2 + 14x + 48 by (x + 6)', 'polynomial division by factorisation', 0.0, 2.0,
            steps=[
                _step(1, 'x + 6', 'incorrect', 0.0, 'The divisor itself was repeated as the quotient instead of taking the other factor.'),
            ],
            scheme=[
                _scheme(1, 'x + 8', 2.0, 'Complete the algebra accurately and state the final result'),
            ],
            feedback=[
                _fb(1, 'You identified the relevant algebra topic and attempted a solution.',
                    missing='The divisor itself was repeated as the quotient instead of taking the other factor. The correct result is x + 8.',
                    deduction='Full marks lost because the stated result does not follow the required algebraic rule.',
                    improve='Once the dividend is written as divisor × factor, cancel the divisor and state the remaining factor.'),
            ],
            topic='polynomial_division', difficulty='difficult',
        ),
        _example(
            'Find the remainder when x^2 + 2x + 3 is divided by (x - 1)', 'remainder theorem', 2.0, 2.0,
            steps=[
                _step(1, '6', 'correct', 2.0),
            ],
            scheme=[
                _scheme(1, '6', 2.0, 'Complete the algebra accurately and state the final result'),
            ],
            feedback=[
                _fb(1, 'You correctly substituted the zero of the divisor into the polynomial and evaluated it.',
                    improve='Set the divisor equal to zero to find the substitution value, then evaluate every term carefully.'),
            ],
            topic='remainder_factor_theorem', difficulty='easy',
        ),
        _example(
            'Find the remainder when x^2 + 3x + 4 is divided by (x - 2)', 'remainder theorem', 1.0, 2.0,
            steps=[
                _step(1, 'f(2) = 4 + 6 + 4', 'partial', 1.0, 'The correct substitution was written, but the numerical remainder was not evaluated.'),
            ],
            scheme=[
                _scheme(1, '14', 2.0, 'Complete the algebra accurately and state the final result'),
            ],
            feedback=[
                _fb(1, 'You made a valid start and used part of the correct algebraic method.',
                    missing='The correct substitution was written, but the numerical remainder was not evaluated. The complete result is 14.',
                    deduction='1 mark deducted because the response is incomplete or not fully simplified.',
                    improve='Set the divisor equal to zero to find the substitution value, then evaluate every term carefully.'),
            ],
            topic='remainder_factor_theorem', difficulty='medium',
        ),
        _example(
            'Find the remainder when x^2 + 4x + 5 is divided by (x - 3)', 'remainder theorem', 1.0, 2.0,
            steps=[
                _step(1, 'f(3) = 9 + 12 + 5', 'partial', 1.0, 'The correct substitution was written, but the numerical remainder was not evaluated.'),
            ],
            scheme=[
                _scheme(1, '26', 2.0, 'Complete the algebra accurately and state the final result'),
            ],
            feedback=[
                _fb(1, 'You made a valid start and used part of the correct algebraic method.',
                    missing='The correct substitution was written, but the numerical remainder was not evaluated. The complete result is 26.',
                    deduction='1 mark deducted because the response is incomplete or not fully simplified.',
                    improve='Set the divisor equal to zero to find the substitution value, then evaluate every term carefully.'),
            ],
            topic='remainder_factor_theorem', difficulty='difficult',
        ),
        _example(
            'Find the remainder when x^2 + 5x + 6 is divided by (x - 4)', 'remainder theorem', 0.0, 2.0,
            steps=[
                _step(1, '2', 'incorrect', 0.0, 'For divisor (x-c), substitute x=c; changing the sign of the linear term gives the wrong evaluation.'),
            ],
            scheme=[
                _scheme(1, '42', 2.0, 'Complete the algebra accurately and state the final result'),
            ],
            feedback=[
                _fb(1, 'You identified the relevant algebra topic and attempted a solution.',
                    missing='For divisor (x-c), substitute x=c; changing the sign of the linear term gives the wrong evaluation. The correct result is 42.',
                    deduction='Full marks lost because the stated result does not follow the required algebraic rule.',
                    improve='Set the divisor equal to zero to find the substitution value, then evaluate every term carefully.'),
            ],
            topic='remainder_factor_theorem', difficulty='medium',
        ),
        _example(
            'Find the remainder when x^2 + 6x + 7 is divided by (x - 5)', 'remainder theorem', 0.0, 2.0,
            steps=[
                _step(1, '2', 'incorrect', 0.0, 'For divisor (x-c), substitute x=c; changing the sign of the linear term gives the wrong evaluation.'),
            ],
            scheme=[
                _scheme(1, '62', 2.0, 'Complete the algebra accurately and state the final result'),
            ],
            feedback=[
                _fb(1, 'You identified the relevant algebra topic and attempted a solution.',
                    missing='For divisor (x-c), substitute x=c; changing the sign of the linear term gives the wrong evaluation. The correct result is 62.',
                    deduction='Full marks lost because the stated result does not follow the required algebraic rule.',
                    improve='Set the divisor equal to zero to find the substitution value, then evaluate every term carefully.'),
            ],
            topic='remainder_factor_theorem', difficulty='difficult',
        ),
        _example(
            'Simplify (x^4 × x^3) / x^1', 'index laws', 2.0, 2.0,
            steps=[
                _step(1, 'x^6', 'correct', 2.0),
            ],
            scheme=[
                _scheme(1, 'x^6', 2.0, 'Complete the algebra accurately and state the final result'),
            ],
            feedback=[
                _fb(1, "You correctly added powers for multiplication and subtracted the divisor's power.",
                    improve='Apply multiplication first (add exponents), then division (subtract the denominator exponent).'),
            ],
            topic='index_laws', difficulty='easy',
        ),
        _example(
            'Simplify (x^5 × x^4) / x^2', 'index laws', 1.0, 2.0,
            steps=[
                _step(1, 'x^(5+4-2)', 'partial', 1.0, 'The correct exponent operation was written, but the exponent was not simplified.'),
            ],
            scheme=[
                _scheme(1, 'x^7', 2.0, 'Complete the algebra accurately and state the final result'),
            ],
            feedback=[
                _fb(1, 'You made a valid start and used part of the correct algebraic method.',
                    missing='The correct exponent operation was written, but the exponent was not simplified. The complete result is x^7.',
                    deduction='1 mark deducted because the response is incomplete or not fully simplified.',
                    improve='Apply multiplication first (add exponents), then division (subtract the denominator exponent).'),
            ],
            topic='index_laws', difficulty='medium',
        ),
        _example(
            'Simplify (x^6 × x^5) / x^3', 'index laws', 1.0, 2.0,
            steps=[
                _step(1, 'x^(6+5-3)', 'partial', 1.0, 'The correct exponent operation was written, but the exponent was not simplified.'),
            ],
            scheme=[
                _scheme(1, 'x^8', 2.0, 'Complete the algebra accurately and state the final result'),
            ],
            feedback=[
                _fb(1, 'You made a valid start and used part of the correct algebraic method.',
                    missing='The correct exponent operation was written, but the exponent was not simplified. The complete result is x^8.',
                    deduction='1 mark deducted because the response is incomplete or not fully simplified.',
                    improve='Apply multiplication first (add exponents), then division (subtract the denominator exponent).'),
            ],
            topic='index_laws', difficulty='difficult',
        ),
        _example(
            'Simplify (x^7 × x^6) / x^4', 'index laws', 0.0, 2.0,
            steps=[
                _step(1, 'x^-3', 'incorrect', 0.0, 'Powers of the same base are added when multiplying; they are not subtracted at that stage.'),
            ],
            scheme=[
                _scheme(1, 'x^9', 2.0, 'Complete the algebra accurately and state the final result'),
            ],
            feedback=[
                _fb(1, 'You identified the relevant algebra topic and attempted a solution.',
                    missing='Powers of the same base are added when multiplying; they are not subtracted at that stage. The correct result is x^9.',
                    deduction='Full marks lost because the stated result does not follow the required algebraic rule.',
                    improve='Apply multiplication first (add exponents), then division (subtract the denominator exponent).'),
            ],
            topic='index_laws', difficulty='medium',
        ),
        _example(
            'Simplify (x^8 × x^7) / x^5', 'index laws', 0.0, 2.0,
            steps=[
                _step(1, 'x^-4', 'incorrect', 0.0, 'Powers of the same base are added when multiplying; they are not subtracted at that stage.'),
            ],
            scheme=[
                _scheme(1, 'x^10', 2.0, 'Complete the algebra accurately and state the final result'),
            ],
            feedback=[
                _fb(1, 'You identified the relevant algebra topic and attempted a solution.',
                    missing='Powers of the same base are added when multiplying; they are not subtracted at that stage. The correct result is x^10.',
                    deduction='Full marks lost because the stated result does not follow the required algebraic rule.',
                    improve='Apply multiplication first (add exponents), then division (subtract the denominator exponent).'),
            ],
            topic='index_laws', difficulty='difficult',
        ),
        _example(
            'Evaluate 16^(-1/2)', 'negative fractional indices', 2.0, 2.0,
            steps=[
                _step(1, '1/4', 'correct', 2.0),
            ],
            scheme=[
                _scheme(1, '1/4', 2.0, 'Complete the algebra accurately and state the final result'),
            ],
            feedback=[
                _fb(1, 'You correctly applied the root, power, and reciprocal indicated by the exponent.',
                    improve='Use a^(−m/n)=1/(n-th root of a)^m and complete the operations in that order.'),
            ],
            topic='negative_fractional_indices', difficulty='easy',
        ),
        _example(
            'Evaluate 125^(-2/3)', 'negative fractional indices', 1.0, 2.0,
            steps=[
                _step(1, '125^(1/3) = 5', 'partial', 1.0, 'The required root was found, but the numerator power and negative-exponent reciprocal were not completed.'),
            ],
            scheme=[
                _scheme(1, '1/25', 2.0, 'Complete the algebra accurately and state the final result'),
            ],
            feedback=[
                _fb(1, 'You made a valid start and used part of the correct algebraic method.',
                    missing='The required root was found, but the numerator power and negative-exponent reciprocal were not completed. The complete result is 1/25.',
                    deduction='1 mark deducted because the response is incomplete or not fully simplified.',
                    improve='Use a^(−m/n)=1/(n-th root of a)^m and complete the operations in that order.'),
            ],
            topic='negative_fractional_indices', difficulty='medium',
        ),
        _example(
            'Evaluate 36^(-3/2)', 'negative fractional indices', 1.0, 2.0,
            steps=[
                _step(1, '36^(1/2) = 6', 'partial', 1.0, 'The required root was found, but the numerator power and negative-exponent reciprocal were not completed.'),
            ],
            scheme=[
                _scheme(1, '1/216', 2.0, 'Complete the algebra accurately and state the final result'),
            ],
            feedback=[
                _fb(1, 'You made a valid start and used part of the correct algebraic method.',
                    missing='The required root was found, but the numerator power and negative-exponent reciprocal were not completed. The complete result is 1/216.',
                    deduction='1 mark deducted because the response is incomplete or not fully simplified.',
                    improve='Use a^(−m/n)=1/(n-th root of a)^m and complete the operations in that order.'),
            ],
            topic='negative_fractional_indices', difficulty='difficult',
        ),
        _example(
            'Evaluate 343^(-1/3)', 'negative fractional indices', 0.0, 2.0,
            steps=[
                _step(1, '7', 'incorrect', 0.0, 'The negative exponent was ignored; it requires taking the reciprocal of the positive-power result.'),
            ],
            scheme=[
                _scheme(1, '1/7', 2.0, 'Complete the algebra accurately and state the final result'),
            ],
            feedback=[
                _fb(1, 'You identified the relevant algebra topic and attempted a solution.',
                    missing='The negative exponent was ignored; it requires taking the reciprocal of the positive-power result. The correct result is 1/7.',
                    deduction='Full marks lost because the stated result does not follow the required algebraic rule.',
                    improve='Use a^(−m/n)=1/(n-th root of a)^m and complete the operations in that order.'),
            ],
            topic='negative_fractional_indices', difficulty='medium',
        ),
        _example(
            'Evaluate 64^(-2/2)', 'negative fractional indices', 0.0, 2.0,
            steps=[
                _step(1, '64', 'incorrect', 0.0, 'The negative exponent was ignored; it requires taking the reciprocal of the positive-power result.'),
            ],
            scheme=[
                _scheme(1, '1/64', 2.0, 'Complete the algebra accurately and state the final result'),
            ],
            feedback=[
                _fb(1, 'You identified the relevant algebra topic and attempted a solution.',
                    missing='The negative exponent was ignored; it requires taking the reciprocal of the positive-power result. The correct result is 1/64.',
                    deduction='Full marks lost because the stated result does not follow the required algebraic rule.',
                    improve='Use a^(−m/n)=1/(n-th root of a)^m and complete the operations in that order.'),
            ],
            topic='negative_fractional_indices', difficulty='difficult',
        ),
        _example(
            'Simplify √18', 'simplifying surds', 2.0, 2.0,
            steps=[
                _step(1, '3√2', 'correct', 2.0),
            ],
            scheme=[
                _scheme(1, '3√2', 2.0, 'Complete the algebra accurately and state the final result'),
            ],
            feedback=[
                _fb(1, 'You correctly extracted the largest perfect-square factor from the radical.',
                    improve='Write the radicand as perfect square × square-free part, then take the perfect square outside the root.'),
            ],
            topic='surds', difficulty='easy',
        ),
        _example(
            'Simplify √48', 'simplifying surds', 1.0, 2.0,
            steps=[
                _step(1, '√(16 × 3)', 'partial', 1.0, 'The correct perfect-square factor was identified, but its square root was not taken outside.'),
            ],
            scheme=[
                _scheme(1, '4√3', 2.0, 'Complete the algebra accurately and state the final result'),
            ],
            feedback=[
                _fb(1, 'You made a valid start and used part of the correct algebraic method.',
                    missing='The correct perfect-square factor was identified, but its square root was not taken outside. The complete result is 4√3.',
                    deduction='1 mark deducted because the response is incomplete or not fully simplified.',
                    improve='Write the radicand as perfect square × square-free part, then take the perfect square outside the root.'),
            ],
            topic='surds', difficulty='medium',
        ),
        _example(
            'Simplify √125', 'simplifying surds', 1.0, 2.0,
            steps=[
                _step(1, '√(25 × 5)', 'partial', 1.0, 'The correct perfect-square factor was identified, but its square root was not taken outside.'),
            ],
            scheme=[
                _scheme(1, '5√5', 2.0, 'Complete the algebra accurately and state the final result'),
            ],
            feedback=[
                _fb(1, 'You made a valid start and used part of the correct algebraic method.',
                    missing='The correct perfect-square factor was identified, but its square root was not taken outside. The complete result is 5√5.',
                    deduction='1 mark deducted because the response is incomplete or not fully simplified.',
                    improve='Write the radicand as perfect square × square-free part, then take the perfect square outside the root.'),
            ],
            topic='surds', difficulty='difficult',
        ),
        _example(
            'Simplify √180', 'simplifying surds', 0.0, 2.0,
            steps=[
                _step(1, '11', 'incorrect', 0.0, 'A square root of a product cannot be replaced by adding the square root factor and the remaining factor.'),
            ],
            scheme=[
                _scheme(1, '6√5', 2.0, 'Complete the algebra accurately and state the final result'),
            ],
            feedback=[
                _fb(1, 'You identified the relevant algebra topic and attempted a solution.',
                    missing='A square root of a product cannot be replaced by adding the square root factor and the remaining factor. The correct result is 6√5.',
                    deduction='Full marks lost because the stated result does not follow the required algebraic rule.',
                    improve='Write the radicand as perfect square × square-free part, then take the perfect square outside the root.'),
            ],
            topic='surds', difficulty='medium',
        ),
        _example(
            'Simplify √98', 'simplifying surds', 0.0, 2.0,
            steps=[
                _step(1, '9', 'incorrect', 0.0, 'A square root of a product cannot be replaced by adding the square root factor and the remaining factor.'),
            ],
            scheme=[
                _scheme(1, '7√2', 2.0, 'Complete the algebra accurately and state the final result'),
            ],
            feedback=[
                _fb(1, 'You identified the relevant algebra topic and attempted a solution.',
                    missing='A square root of a product cannot be replaced by adding the square root factor and the remaining factor. The correct result is 7√2.',
                    deduction='Full marks lost because the stated result does not follow the required algebraic rule.',
                    improve='Write the radicand as perfect square × square-free part, then take the perfect square outside the root.'),
            ],
            topic='surds', difficulty='difficult',
        ),
        _example(
            'Rationalise 2/√2', 'rationalising a surd denominator', 2.0, 2.0,
            steps=[
                _step(1, '√2', 'correct', 2.0),
            ],
            scheme=[
                _scheme(1, '√2', 2.0, 'Complete the algebra accurately and state the final result'),
            ],
            feedback=[
                _fb(1, 'You correctly multiplied the whole fraction by √m/√m and simplified the denominator.',
                    improve="Multiply both numerator and denominator by the denominator's surd, then simplify both parts."),
            ],
            topic='rationalising_denominators', difficulty='easy',
        ),
        _example(
            'Rationalise 3/√3', 'rationalising a surd denominator', 1.0, 2.0,
            steps=[
                _step(1, '(3√3)/(√3×√3)', 'partial', 1.0, 'The correct rationalising multiplication was set up, but the denominator and final fraction were not simplified.'),
            ],
            scheme=[
                _scheme(1, '√3', 2.0, 'Complete the algebra accurately and state the final result'),
            ],
            feedback=[
                _fb(1, 'You made a valid start and used part of the correct algebraic method.',
                    missing='The correct rationalising multiplication was set up, but the denominator and final fraction were not simplified. The complete result is √3.',
                    deduction='1 mark deducted because the response is incomplete or not fully simplified.',
                    improve="Multiply both numerator and denominator by the denominator's surd, then simplify both parts."),
            ],
            topic='rationalising_denominators', difficulty='medium',
        ),
        _example(
            'Rationalise 4/√5', 'rationalising a surd denominator', 1.0, 2.0,
            steps=[
                _step(1, '(4√5)/(√5×√5)', 'partial', 1.0, 'The correct rationalising multiplication was set up, but the denominator and final fraction were not simplified.'),
            ],
            scheme=[
                _scheme(1, '4√5/5', 2.0, 'Complete the algebra accurately and state the final result'),
            ],
            feedback=[
                _fb(1, 'You made a valid start and used part of the correct algebraic method.',
                    missing='The correct rationalising multiplication was set up, but the denominator and final fraction were not simplified. The complete result is 4√5/5.',
                    deduction='1 mark deducted because the response is incomplete or not fully simplified.',
                    improve="Multiply both numerator and denominator by the denominator's surd, then simplify both parts."),
            ],
            topic='rationalising_denominators', difficulty='difficult',
        ),
        _example(
            'Rationalise 5/√6', 'rationalising a surd denominator', 0.0, 2.0,
            steps=[
                _step(1, '5/6', 'incorrect', 0.0, 'The numerator must also be multiplied by the surd; otherwise the value of the fraction changes.'),
            ],
            scheme=[
                _scheme(1, '5√6/6', 2.0, 'Complete the algebra accurately and state the final result'),
            ],
            feedback=[
                _fb(1, 'You identified the relevant algebra topic and attempted a solution.',
                    missing='The numerator must also be multiplied by the surd; otherwise the value of the fraction changes. The correct result is 5√6/6.',
                    deduction='Full marks lost because the stated result does not follow the required algebraic rule.',
                    improve="Multiply both numerator and denominator by the denominator's surd, then simplify both parts."),
            ],
            topic='rationalising_denominators', difficulty='medium',
        ),
        _example(
            'Rationalise 6/√7', 'rationalising a surd denominator', 0.0, 2.0,
            steps=[
                _step(1, '6/7', 'incorrect', 0.0, 'The numerator must also be multiplied by the surd; otherwise the value of the fraction changes.'),
            ],
            scheme=[
                _scheme(1, '6√7/7', 2.0, 'Complete the algebra accurately and state the final result'),
            ],
            feedback=[
                _fb(1, 'You identified the relevant algebra topic and attempted a solution.',
                    missing='The numerator must also be multiplied by the surd; otherwise the value of the fraction changes. The correct result is 6√7/7.',
                    deduction='Full marks lost because the stated result does not follow the required algebraic rule.',
                    improve="Multiply both numerator and denominator by the denominator's surd, then simplify both parts."),
            ],
            topic='rationalising_denominators', difficulty='difficult',
        ),
        _example(
            'If f(x) = 1x^2 + 2x + 3, find f(-2)', 'function evaluation', 2.0, 2.0,
            steps=[
                _step(1, '3', 'correct', 2.0),
            ],
            scheme=[
                _scheme(1, '3', 2.0, 'Complete the algebra accurately and state the final result'),
            ],
            feedback=[
                _fb(1, 'You correctly substituted the input and followed the order of operations.',
                    improve='Substitute using brackets, evaluate powers first, then multiplication, addition, and subtraction.'),
            ],
            topic='functions', difficulty='easy',
        ),
        _example(
            'If f(x) = 2x^2 + 3x + 4, find f(-1)', 'function evaluation', 1.0, 2.0,
            steps=[
                _step(1, 'f(-1) = 2(-1)^2 + 3(-1) + 4', 'partial', 1.0, 'The substitution is correct, but the arithmetic was not evaluated to a final value.'),
            ],
            scheme=[
                _scheme(1, '3', 2.0, 'Complete the algebra accurately and state the final result'),
            ],
            feedback=[
                _fb(1, 'You made a valid start and used part of the correct algebraic method.',
                    missing='The substitution is correct, but the arithmetic was not evaluated to a final value. The complete result is 3.',
                    deduction='1 mark deducted because the response is incomplete or not fully simplified.',
                    improve='Substitute using brackets, evaluate powers first, then multiplication, addition, and subtraction.'),
            ],
            topic='functions', difficulty='medium',
        ),
        _example(
            'If f(x) = 3x^2 + 4x + 5, find f(0)', 'function evaluation', 1.0, 2.0,
            steps=[
                _step(1, 'f(0) = 3(0)^2 + 4(0) + 5', 'partial', 1.0, 'The substitution is correct, but the arithmetic was not evaluated to a final value.'),
            ],
            scheme=[
                _scheme(1, '5', 2.0, 'Complete the algebra accurately and state the final result'),
            ],
            feedback=[
                _fb(1, 'You made a valid start and used part of the correct algebraic method.',
                    missing='The substitution is correct, but the arithmetic was not evaluated to a final value. The complete result is 5.',
                    deduction='1 mark deducted because the response is incomplete or not fully simplified.',
                    improve='Substitute using brackets, evaluate powers first, then multiplication, addition, and subtraction.'),
            ],
            topic='functions', difficulty='difficult',
        ),
        _example(
            'If f(x) = 4x^2 + 5x + 6, find f(1)', 'function evaluation', 0.0, 2.0,
            steps=[
                _step(1, '7', 'incorrect', 0.0, 'The squared term was given the wrong sign; a negative input squared is positive.'),
            ],
            scheme=[
                _scheme(1, '15', 2.0, 'Complete the algebra accurately and state the final result'),
            ],
            feedback=[
                _fb(1, 'You identified the relevant algebra topic and attempted a solution.',
                    missing='The squared term was given the wrong sign; a negative input squared is positive. The correct result is 15.',
                    deduction='Full marks lost because the stated result does not follow the required algebraic rule.',
                    improve='Substitute using brackets, evaluate powers first, then multiplication, addition, and subtraction.'),
            ],
            topic='functions', difficulty='medium',
        ),
        _example(
            'If f(x) = 5x^2 + 6x + 7, find f(2)', 'function evaluation', 0.0, 2.0,
            steps=[
                _step(1, '-1', 'incorrect', 0.0, 'The squared term was given the wrong sign; a negative input squared is positive.'),
            ],
            scheme=[
                _scheme(1, '39', 2.0, 'Complete the algebra accurately and state the final result'),
            ],
            feedback=[
                _fb(1, 'You identified the relevant algebra topic and attempted a solution.',
                    missing='The squared term was given the wrong sign; a negative input squared is positive. The correct result is 39.',
                    deduction='Full marks lost because the stated result does not follow the required algebraic rule.',
                    improve='Substitute using brackets, evaluate powers first, then multiplication, addition, and subtraction.'),
            ],
            topic='functions', difficulty='difficult',
        ),
        _example(
            'If f(x) = 2x + 1 and g(x) = x^2 + 3, find (f∘g)(1)', 'function composition', 2.0, 2.0,
            steps=[
                _step(1, '9', 'correct', 2.0),
            ],
            scheme=[
                _scheme(1, '9', 2.0, 'Complete the algebra accurately and state the final result'),
            ],
            feedback=[
                _fb(1, 'You correctly evaluated the inner function first and used its result in the outer function.',
                    improve='Read composition from right to left: evaluate g first, then apply f to that result.'),
            ],
            topic='function_composition', difficulty='easy',
        ),
        _example(
            'If f(x) = 3x + 2 and g(x) = x^2 + 4, find (f∘g)(2)', 'function composition', 1.0, 2.0,
            steps=[
                _step(1, 'g(2) = 8', 'partial', 1.0, 'The inner function was evaluated correctly, but the result was not substituted into f.'),
            ],
            scheme=[
                _scheme(1, '26', 2.0, 'Complete the algebra accurately and state the final result'),
            ],
            feedback=[
                _fb(1, 'You made a valid start and used part of the correct algebraic method.',
                    missing='The inner function was evaluated correctly, but the result was not substituted into f. The complete result is 26.',
                    deduction='1 mark deducted because the response is incomplete or not fully simplified.',
                    improve='Read composition from right to left: evaluate g first, then apply f to that result.'),
            ],
            topic='function_composition', difficulty='medium',
        ),
        _example(
            'If f(x) = 4x + 3 and g(x) = x^2 + 5, find (f∘g)(3)', 'function composition', 1.0, 2.0,
            steps=[
                _step(1, 'g(3) = 14', 'partial', 1.0, 'The inner function was evaluated correctly, but the result was not substituted into f.'),
            ],
            scheme=[
                _scheme(1, '59', 2.0, 'Complete the algebra accurately and state the final result'),
            ],
            feedback=[
                _fb(1, 'You made a valid start and used part of the correct algebraic method.',
                    missing='The inner function was evaluated correctly, but the result was not substituted into f. The complete result is 59.',
                    deduction='1 mark deducted because the response is incomplete or not fully simplified.',
                    improve='Read composition from right to left: evaluate g first, then apply f to that result.'),
            ],
            topic='function_composition', difficulty='difficult',
        ),
        _example(
            'If f(x) = 5x + 4 and g(x) = x^2 + 6, find (f∘g)(4)', 'function composition', 0.0, 2.0,
            steps=[
                _step(1, 'g(f(4)) = 582', 'incorrect', 0.0, 'The composition order was reversed; f∘g means f(g(x)), not g(f(x)).'),
            ],
            scheme=[
                _scheme(1, '114', 2.0, 'Complete the algebra accurately and state the final result'),
            ],
            feedback=[
                _fb(1, 'You identified the relevant algebra topic and attempted a solution.',
                    missing='The composition order was reversed; f∘g means f(g(x)), not g(f(x)). The correct result is 114.',
                    deduction='Full marks lost because the stated result does not follow the required algebraic rule.',
                    improve='Read composition from right to left: evaluate g first, then apply f to that result.'),
            ],
            topic='function_composition', difficulty='medium',
        ),
        _example(
            'If f(x) = 6x + 5 and g(x) = x^2 + 7, find (f∘g)(5)', 'function composition', 0.0, 2.0,
            steps=[
                _step(1, 'g(f(5)) = 1232', 'incorrect', 0.0, 'The composition order was reversed; f∘g means f(g(x)), not g(f(x)).'),
            ],
            scheme=[
                _scheme(1, '197', 2.0, 'Complete the algebra accurately and state the final result'),
            ],
            feedback=[
                _fb(1, 'You identified the relevant algebra topic and attempted a solution.',
                    missing='The composition order was reversed; f∘g means f(g(x)), not g(f(x)). The correct result is 197.',
                    deduction='Full marks lost because the stated result does not follow the required algebraic rule.',
                    improve='Read composition from right to left: evaluate g first, then apply f to that result.'),
            ],
            topic='function_composition', difficulty='difficult',
        ),
        _example(
            'Find the inverse of f(x) = 2x + 3', 'inverse function', 2.0, 2.0,
            steps=[
                _step(1, 'f^-1(x) = (x - 3)/2', 'correct', 2.0),
            ],
            scheme=[
                _scheme(1, 'f^-1(x) = (x - 3)/2', 2.0, 'Complete the algebra accurately and state the final result'),
            ],
            feedback=[
                _fb(1, 'You correctly swapped x and y and solved the new equation for y.',
                    improve='Write y=f(x), swap x and y, isolate y using inverse operations, then rename it f^-1(x).'),
            ],
            topic='inverse_functions', difficulty='easy',
        ),
        _example(
            'Find the inverse of f(x) = 3x + 4', 'inverse function', 1.0, 2.0,
            steps=[
                _step(1, 'x = 3y + 4', 'partial', 1.0, 'x and y were swapped correctly, but the equation was not solved for y.'),
            ],
            scheme=[
                _scheme(1, 'f^-1(x) = (x - 4)/3', 2.0, 'Complete the algebra accurately and state the final result'),
            ],
            feedback=[
                _fb(1, 'You made a valid start and used part of the correct algebraic method.',
                    missing='x and y were swapped correctly, but the equation was not solved for y. The complete result is f^-1(x) = (x - 4)/3.',
                    deduction='1 mark deducted because the response is incomplete or not fully simplified.',
                    improve='Write y=f(x), swap x and y, isolate y using inverse operations, then rename it f^-1(x).'),
            ],
            topic='inverse_functions', difficulty='medium',
        ),
        _example(
            'Find the inverse of f(x) = 4x + 5', 'inverse function', 1.0, 2.0,
            steps=[
                _step(1, 'x = 4y + 5', 'partial', 1.0, 'x and y were swapped correctly, but the equation was not solved for y.'),
            ],
            scheme=[
                _scheme(1, 'f^-1(x) = (x - 5)/4', 2.0, 'Complete the algebra accurately and state the final result'),
            ],
            feedback=[
                _fb(1, 'You made a valid start and used part of the correct algebraic method.',
                    missing='x and y were swapped correctly, but the equation was not solved for y. The complete result is f^-1(x) = (x - 5)/4.',
                    deduction='1 mark deducted because the response is incomplete or not fully simplified.',
                    improve='Write y=f(x), swap x and y, isolate y using inverse operations, then rename it f^-1(x).'),
            ],
            topic='inverse_functions', difficulty='difficult',
        ),
        _example(
            'Find the inverse of f(x) = 5x + 6', 'inverse function', 0.0, 2.0,
            steps=[
                _step(1, 'f^-1(x) = (x + 6)/5', 'incorrect', 0.0, 'To undo +b, subtract b; adding b again gives the wrong inverse.'),
            ],
            scheme=[
                _scheme(1, 'f^-1(x) = (x - 6)/5', 2.0, 'Complete the algebra accurately and state the final result'),
            ],
            feedback=[
                _fb(1, 'You identified the relevant algebra topic and attempted a solution.',
                    missing='To undo +b, subtract b; adding b again gives the wrong inverse. The correct result is f^-1(x) = (x - 6)/5.',
                    deduction='Full marks lost because the stated result does not follow the required algebraic rule.',
                    improve='Write y=f(x), swap x and y, isolate y using inverse operations, then rename it f^-1(x).'),
            ],
            topic='inverse_functions', difficulty='medium',
        ),
        _example(
            'Find the inverse of f(x) = 6x + 7', 'inverse function', 0.0, 2.0,
            steps=[
                _step(1, 'f^-1(x) = (x + 7)/6', 'incorrect', 0.0, 'To undo +b, subtract b; adding b again gives the wrong inverse.'),
            ],
            scheme=[
                _scheme(1, 'f^-1(x) = (x - 7)/6', 2.0, 'Complete the algebra accurately and state the final result'),
            ],
            feedback=[
                _fb(1, 'You identified the relevant algebra topic and attempted a solution.',
                    missing='To undo +b, subtract b; adding b again gives the wrong inverse. The correct result is f^-1(x) = (x - 7)/6.',
                    deduction='Full marks lost because the stated result does not follow the required algebraic rule.',
                    improve='Write y=f(x), swap x and y, isolate y using inverse operations, then rename it f^-1(x).'),
            ],
            topic='inverse_functions', difficulty='difficult',
        ),
        _example(
            'Find the equation of the line with gradient 2 through (1, 3)', 'point-gradient form', 2.0, 2.0,
            steps=[
                _step(1, 'y = 2x + 1', 'correct', 2.0),
            ],
            scheme=[
                _scheme(1, 'y = 2x + 1', 2.0, 'Complete the algebra accurately and state the final result'),
            ],
            feedback=[
                _fb(1, 'You correctly used point-gradient form and rearranged to y=mx+c.',
                    improve='Substitute the point into y=mx+c or expand point-gradient form, then solve carefully for c.'),
            ],
            topic='straight_line_equations', difficulty='easy',
        ),
        _example(
            'Find the equation of the line with gradient 3 through (2, 5)', 'point-gradient form', 1.0, 2.0,
            steps=[
                _step(1, 'y - 5 = 3(x - 2)', 'partial', 1.0, 'The point-gradient equation is correct, but it was not expanded and rearranged to final form.'),
            ],
            scheme=[
                _scheme(1, 'y = 3x - 1', 2.0, 'Complete the algebra accurately and state the final result'),
            ],
            feedback=[
                _fb(1, 'You made a valid start and used part of the correct algebraic method.',
                    missing='The point-gradient equation is correct, but it was not expanded and rearranged to final form. The complete result is y = 3x - 1.',
                    deduction='1 mark deducted because the response is incomplete or not fully simplified.',
                    improve='Substitute the point into y=mx+c or expand point-gradient form, then solve carefully for c.'),
            ],
            topic='straight_line_equations', difficulty='medium',
        ),
        _example(
            'Find the equation of the line with gradient 4 through (3, 7)', 'point-gradient form', 1.0, 2.0,
            steps=[
                _step(1, 'y - 7 = 4(x - 3)', 'partial', 1.0, 'The point-gradient equation is correct, but it was not expanded and rearranged to final form.'),
            ],
            scheme=[
                _scheme(1, 'y = 4x - 5', 2.0, 'Complete the algebra accurately and state the final result'),
            ],
            feedback=[
                _fb(1, 'You made a valid start and used part of the correct algebraic method.',
                    missing='The point-gradient equation is correct, but it was not expanded and rearranged to final form. The complete result is y = 4x - 5.',
                    deduction='1 mark deducted because the response is incomplete or not fully simplified.',
                    improve='Substitute the point into y=mx+c or expand point-gradient form, then solve carefully for c.'),
            ],
            topic='straight_line_equations', difficulty='difficult',
        ),
        _example(
            'Find the equation of the line with gradient 5 through (4, 9)', 'point-gradient form', 0.0, 2.0,
            steps=[
                _step(1, 'y = 5x + 29', 'incorrect', 0.0, 'The intercept was calculated by adding mx to y instead of using c=y-mx.'),
            ],
            scheme=[
                _scheme(1, 'y = 5x - 11', 2.0, 'Complete the algebra accurately and state the final result'),
            ],
            feedback=[
                _fb(1, 'You identified the relevant algebra topic and attempted a solution.',
                    missing='The intercept was calculated by adding mx to y instead of using c=y-mx. The correct result is y = 5x - 11.',
                    deduction='Full marks lost because the stated result does not follow the required algebraic rule.',
                    improve='Substitute the point into y=mx+c or expand point-gradient form, then solve carefully for c.'),
            ],
            topic='straight_line_equations', difficulty='medium',
        ),
        _example(
            'Find the equation of the line with gradient 6 through (5, 11)', 'point-gradient form', 0.0, 2.0,
            steps=[
                _step(1, 'y = 6x + 41', 'incorrect', 0.0, 'The intercept was calculated by adding mx to y instead of using c=y-mx.'),
            ],
            scheme=[
                _scheme(1, 'y = 6x - 19', 2.0, 'Complete the algebra accurately and state the final result'),
            ],
            feedback=[
                _fb(1, 'You identified the relevant algebra topic and attempted a solution.',
                    missing='The intercept was calculated by adding mx to y instead of using c=y-mx. The correct result is y = 6x - 19.',
                    deduction='Full marks lost because the stated result does not follow the required algebraic rule.',
                    improve='Substitute the point into y=mx+c or expand point-gradient form, then solve carefully for c.'),
            ],
            topic='straight_line_equations', difficulty='difficult',
        ),
        _example(
            'State the gradient and y-intercept of -2x + 2y = 4', 'rearranging to y=mx+c', 2.0, 2.0,
            steps=[
                _step(1, 'gradient = 1, y-intercept = 2', 'correct', 2.0),
            ],
            scheme=[
                _scheme(1, 'gradient = 1, y-intercept = 2', 2.0, 'Complete the algebra accurately and state the final result'),
            ],
            feedback=[
                _fb(1, 'You correctly rearranged the equation to y=mx+c and read both values.',
                    improve='Isolate By, divide the entire equation by B, and read m and c only after reaching y=mx+c.'),
            ],
            topic='gradient_intercept', difficulty='easy',
        ),
        _example(
            'State the gradient and y-intercept of -6x + 3y = 9', 'rearranging to y=mx+c', 1.0, 2.0,
            steps=[
                _step(1, '3y = 6x + 9', 'partial', 1.0, 'The x term was moved correctly, but every term was not divided by the coefficient of y.'),
            ],
            scheme=[
                _scheme(1, 'gradient = 2, y-intercept = 3', 2.0, 'Complete the algebra accurately and state the final result'),
            ],
            feedback=[
                _fb(1, 'You made a valid start and used part of the correct algebraic method.',
                    missing='The x term was moved correctly, but every term was not divided by the coefficient of y. The complete result is gradient = 2, y-intercept = 3.',
                    deduction='1 mark deducted because the response is incomplete or not fully simplified.',
                    improve='Isolate By, divide the entire equation by B, and read m and c only after reaching y=mx+c.'),
            ],
            topic='gradient_intercept', difficulty='medium',
        ),
        _example(
            'State the gradient and y-intercept of -12x + 4y = 16', 'rearranging to y=mx+c', 1.0, 2.0,
            steps=[
                _step(1, '4y = 12x + 16', 'partial', 1.0, 'The x term was moved correctly, but every term was not divided by the coefficient of y.'),
            ],
            scheme=[
                _scheme(1, 'gradient = 3, y-intercept = 4', 2.0, 'Complete the algebra accurately and state the final result'),
            ],
            feedback=[
                _fb(1, 'You made a valid start and used part of the correct algebraic method.',
                    missing='The x term was moved correctly, but every term was not divided by the coefficient of y. The complete result is gradient = 3, y-intercept = 4.',
                    deduction='1 mark deducted because the response is incomplete or not fully simplified.',
                    improve='Isolate By, divide the entire equation by B, and read m and c only after reaching y=mx+c.'),
            ],
            topic='gradient_intercept', difficulty='difficult',
        ),
        _example(
            'State the gradient and y-intercept of -20x + 5y = 25', 'rearranging to y=mx+c', 0.0, 2.0,
            steps=[
                _step(1, 'gradient = -4, y-intercept = 25', 'incorrect', 0.0, 'The sign and scaling were not updated after rearranging and dividing by the y coefficient.'),
            ],
            scheme=[
                _scheme(1, 'gradient = 4, y-intercept = 5', 2.0, 'Complete the algebra accurately and state the final result'),
            ],
            feedback=[
                _fb(1, 'You identified the relevant algebra topic and attempted a solution.',
                    missing='The sign and scaling were not updated after rearranging and dividing by the y coefficient. The correct result is gradient = 4, y-intercept = 5.',
                    deduction='Full marks lost because the stated result does not follow the required algebraic rule.',
                    improve='Isolate By, divide the entire equation by B, and read m and c only after reaching y=mx+c.'),
            ],
            topic='gradient_intercept', difficulty='medium',
        ),
        _example(
            'State the gradient and y-intercept of -30x + 6y = 36', 'rearranging to y=mx+c', 0.0, 2.0,
            steps=[
                _step(1, 'gradient = -5, y-intercept = 36', 'incorrect', 0.0, 'The sign and scaling were not updated after rearranging and dividing by the y coefficient.'),
            ],
            scheme=[
                _scheme(1, 'gradient = 5, y-intercept = 6', 2.0, 'Complete the algebra accurately and state the final result'),
            ],
            feedback=[
                _fb(1, 'You identified the relevant algebra topic and attempted a solution.',
                    missing='The sign and scaling were not updated after rearranging and dividing by the y coefficient. The correct result is gradient = 5, y-intercept = 6.',
                    deduction='Full marks lost because the stated result does not follow the required algebraic rule.',
                    improve='Isolate By, divide the entire equation by B, and read m and c only after reaching y=mx+c.'),
            ],
            topic='gradient_intercept', difficulty='difficult',
        ),
        _example(
            'Find the nth term of 4, 6, 8, 10, ...', 'arithmetic nth term', 2.0, 2.0,
            steps=[
                _step(1, '2n + 2', 'correct', 2.0),
            ],
            scheme=[
                _scheme(1, '2n + 2', 2.0, 'Complete the algebra accurately and state the final result'),
            ],
            feedback=[
                _fb(1, 'You correctly used the common difference and adjusted the constant so n=1 gives the first term.',
                    improve='Start with dn, compare its first value d with the actual first term a, and adjust by a-d.'),
            ],
            topic='sequences_nth_term', difficulty='easy',
        ),
        _example(
            'Find the nth term of 5, 8, 11, 14, ...', 'arithmetic nth term', 1.0, 2.0,
            steps=[
                _step(1, 'common difference = 3', 'partial', 1.0, 'The common difference was found, but the nth-term expression was not formed.'),
            ],
            scheme=[
                _scheme(1, '3n + 2', 2.0, 'Complete the algebra accurately and state the final result'),
            ],
            feedback=[
                _fb(1, 'You made a valid start and used part of the correct algebraic method.',
                    missing='The common difference was found, but the nth-term expression was not formed. The complete result is 3n + 2.',
                    deduction='1 mark deducted because the response is incomplete or not fully simplified.',
                    improve='Start with dn, compare its first value d with the actual first term a, and adjust by a-d.'),
            ],
            topic='sequences_nth_term', difficulty='medium',
        ),
        _example(
            'Find the nth term of 6, 10, 14, 18, ...', 'arithmetic nth term', 1.0, 2.0,
            steps=[
                _step(1, 'common difference = 4', 'partial', 1.0, 'The common difference was found, but the nth-term expression was not formed.'),
            ],
            scheme=[
                _scheme(1, '4n + 2', 2.0, 'Complete the algebra accurately and state the final result'),
            ],
            feedback=[
                _fb(1, 'You made a valid start and used part of the correct algebraic method.',
                    missing='The common difference was found, but the nth-term expression was not formed. The complete result is 4n + 2.',
                    deduction='1 mark deducted because the response is incomplete or not fully simplified.',
                    improve='Start with dn, compare its first value d with the actual first term a, and adjust by a-d.'),
            ],
            topic='sequences_nth_term', difficulty='difficult',
        ),
        _example(
            'Find the nth term of 7, 12, 17, 22, ...', 'arithmetic nth term', 0.0, 2.0,
            steps=[
                _step(1, '5n + 7', 'incorrect', 0.0, 'Using the first term as the constant in dn+a makes the n=1 value too large; the constant must be a-d.'),
            ],
            scheme=[
                _scheme(1, '5n + 2', 2.0, 'Complete the algebra accurately and state the final result'),
            ],
            feedback=[
                _fb(1, 'You identified the relevant algebra topic and attempted a solution.',
                    missing='Using the first term as the constant in dn+a makes the n=1 value too large; the constant must be a-d. The correct result is 5n + 2.',
                    deduction='Full marks lost because the stated result does not follow the required algebraic rule.',
                    improve='Start with dn, compare its first value d with the actual first term a, and adjust by a-d.'),
            ],
            topic='sequences_nth_term', difficulty='medium',
        ),
        _example(
            'Find the nth term of 8, 14, 20, 26, ...', 'arithmetic nth term', 0.0, 2.0,
            steps=[
                _step(1, '6n + 8', 'incorrect', 0.0, 'Using the first term as the constant in dn+a makes the n=1 value too large; the constant must be a-d.'),
            ],
            scheme=[
                _scheme(1, '6n + 2', 2.0, 'Complete the algebra accurately and state the final result'),
            ],
            feedback=[
                _fb(1, 'You identified the relevant algebra topic and attempted a solution.',
                    missing='Using the first term as the constant in dn+a makes the n=1 value too large; the constant must be a-d. The correct result is 6n + 2.',
                    deduction='Full marks lost because the stated result does not follow the required algebraic rule.',
                    improve='Start with dn, compare its first value d with the actual first term a, and adjust by a-d.'),
            ],
            topic='sequences_nth_term', difficulty='difficult',
        ),
        _example(
            'An AP has first term 3 and common difference 2. Find the 6th term.', 'arithmetic sequence', 2.0, 2.0,
            steps=[
                _step(1, '13', 'correct', 2.0),
            ],
            scheme=[
                _scheme(1, '13', 2.0, 'Complete the algebra accurately and state the final result'),
            ],
            feedback=[
                _fb(1, 'You correctly applied T_n=a+(n-1)d and evaluated it.',
                    improve='Use T_n=a+(n-1)d, substitute carefully, and evaluate the bracket before multiplying.'),
            ],
            topic='arithmetic_sequences', difficulty='easy',
        ),
        _example(
            'An AP has first term 4 and common difference 3. Find the 7th term.', 'arithmetic sequence', 1.0, 2.0,
            steps=[
                _step(1, 'T_7 = 4 + (7-1)(3)', 'partial', 1.0, 'The correct formula and substitutions were written, but the arithmetic was not evaluated.'),
            ],
            scheme=[
                _scheme(1, '22', 2.0, 'Complete the algebra accurately and state the final result'),
            ],
            feedback=[
                _fb(1, 'You made a valid start and used part of the correct algebraic method.',
                    missing='The correct formula and substitutions were written, but the arithmetic was not evaluated. The complete result is 22.',
                    deduction='1 mark deducted because the response is incomplete or not fully simplified.',
                    improve='Use T_n=a+(n-1)d, substitute carefully, and evaluate the bracket before multiplying.'),
            ],
            topic='arithmetic_sequences', difficulty='medium',
        ),
        _example(
            'An AP has first term 5 and common difference 4. Find the 8th term.', 'arithmetic sequence', 1.0, 2.0,
            steps=[
                _step(1, 'T_8 = 5 + (8-1)(4)', 'partial', 1.0, 'The correct formula and substitutions were written, but the arithmetic was not evaluated.'),
            ],
            scheme=[
                _scheme(1, '33', 2.0, 'Complete the algebra accurately and state the final result'),
            ],
            feedback=[
                _fb(1, 'You made a valid start and used part of the correct algebraic method.',
                    missing='The correct formula and substitutions were written, but the arithmetic was not evaluated. The complete result is 33.',
                    deduction='1 mark deducted because the response is incomplete or not fully simplified.',
                    improve='Use T_n=a+(n-1)d, substitute carefully, and evaluate the bracket before multiplying.'),
            ],
            topic='arithmetic_sequences', difficulty='difficult',
        ),
        _example(
            'An AP has first term 6 and common difference 5. Find the 9th term.', 'arithmetic sequence', 0.0, 2.0,
            steps=[
                _step(1, '51', 'incorrect', 0.0, 'The formula uses n-1 differences from the first term, not n differences.'),
            ],
            scheme=[
                _scheme(1, '46', 2.0, 'Complete the algebra accurately and state the final result'),
            ],
            feedback=[
                _fb(1, 'You identified the relevant algebra topic and attempted a solution.',
                    missing='The formula uses n-1 differences from the first term, not n differences. The correct result is 46.',
                    deduction='Full marks lost because the stated result does not follow the required algebraic rule.',
                    improve='Use T_n=a+(n-1)d, substitute carefully, and evaluate the bracket before multiplying.'),
            ],
            topic='arithmetic_sequences', difficulty='medium',
        ),
        _example(
            'An AP has first term 7 and common difference 6. Find the 10th term.', 'arithmetic sequence', 0.0, 2.0,
            steps=[
                _step(1, '67', 'incorrect', 0.0, 'The formula uses n-1 differences from the first term, not n differences.'),
            ],
            scheme=[
                _scheme(1, '61', 2.0, 'Complete the algebra accurately and state the final result'),
            ],
            feedback=[
                _fb(1, 'You identified the relevant algebra topic and attempted a solution.',
                    missing='The formula uses n-1 differences from the first term, not n differences. The correct result is 61.',
                    deduction='Full marks lost because the stated result does not follow the required algebraic rule.',
                    improve='Use T_n=a+(n-1)d, substitute carefully, and evaluate the bracket before multiplying.'),
            ],
            topic='arithmetic_sequences', difficulty='difficult',
        ),
        _example(
            'A GP has first term 2 and common ratio 2. Find the 4th term.', 'geometric sequence', 2.0, 2.0,
            steps=[
                _step(1, '16', 'correct', 2.0),
            ],
            scheme=[
                _scheme(1, '16', 2.0, 'Complete the algebra accurately and state the final result'),
            ],
            feedback=[
                _fb(1, 'You correctly applied T_n=ar^(n-1) and evaluated the power.',
                    improve='For a GP use repeated multiplication: T_n=ar^(n-1), not a+(n-1)r.'),
            ],
            topic='geometric_sequences', difficulty='easy',
        ),
        _example(
            'A GP has first term 3 and common ratio 3. Find the 5th term.', 'geometric sequence', 1.0, 2.0,
            steps=[
                _step(1, 'T_5 = 3 × 3^4', 'partial', 1.0, 'The correct geometric nth-term expression was set up, but it was not evaluated.'),
            ],
            scheme=[
                _scheme(1, '243', 2.0, 'Complete the algebra accurately and state the final result'),
            ],
            feedback=[
                _fb(1, 'You made a valid start and used part of the correct algebraic method.',
                    missing='The correct geometric nth-term expression was set up, but it was not evaluated. The complete result is 243.',
                    deduction='1 mark deducted because the response is incomplete or not fully simplified.',
                    improve='For a GP use repeated multiplication: T_n=ar^(n-1), not a+(n-1)r.'),
            ],
            topic='geometric_sequences', difficulty='medium',
        ),
        _example(
            'A GP has first term 4 and common ratio 2. Find the 6th term.', 'geometric sequence', 1.0, 2.0,
            steps=[
                _step(1, 'T_6 = 4 × 2^5', 'partial', 1.0, 'The correct geometric nth-term expression was set up, but it was not evaluated.'),
            ],
            scheme=[
                _scheme(1, '128', 2.0, 'Complete the algebra accurately and state the final result'),
            ],
            feedback=[
                _fb(1, 'You made a valid start and used part of the correct algebraic method.',
                    missing='The correct geometric nth-term expression was set up, but it was not evaluated. The complete result is 128.',
                    deduction='1 mark deducted because the response is incomplete or not fully simplified.',
                    improve='For a GP use repeated multiplication: T_n=ar^(n-1), not a+(n-1)r.'),
            ],
            topic='geometric_sequences', difficulty='difficult',
        ),
        _example(
            'A GP has first term 5 and common ratio 3. Find the 7th term.', 'geometric sequence', 0.0, 2.0,
            steps=[
                _step(1, '23', 'incorrect', 0.0, 'An arithmetic addition formula was used even though the sequence changes by a common ratio.'),
            ],
            scheme=[
                _scheme(1, '3645', 2.0, 'Complete the algebra accurately and state the final result'),
            ],
            feedback=[
                _fb(1, 'You identified the relevant algebra topic and attempted a solution.',
                    missing='An arithmetic addition formula was used even though the sequence changes by a common ratio. The correct result is 3645.',
                    deduction='Full marks lost because the stated result does not follow the required algebraic rule.',
                    improve='For a GP use repeated multiplication: T_n=ar^(n-1), not a+(n-1)r.'),
            ],
            topic='geometric_sequences', difficulty='medium',
        ),
        _example(
            'A GP has first term 6 and common ratio 2. Find the 8th term.', 'geometric sequence', 0.0, 2.0,
            steps=[
                _step(1, '20', 'incorrect', 0.0, 'An arithmetic addition formula was used even though the sequence changes by a common ratio.'),
            ],
            scheme=[
                _scheme(1, '768', 2.0, 'Complete the algebra accurately and state the final result'),
            ],
            feedback=[
                _fb(1, 'You identified the relevant algebra topic and attempted a solution.',
                    missing='An arithmetic addition formula was used even though the sequence changes by a common ratio. The correct result is 768.',
                    deduction='Full marks lost because the stated result does not follow the required algebraic rule.',
                    improve='For a GP use repeated multiplication: T_n=ar^(n-1), not a+(n-1)r.'),
            ],
            topic='geometric_sequences', difficulty='difficult',
        ),
        _example(
            'The sum of two consecutive integers is 73. Find the integers.', 'algebraic modelling', 2.0, 2.0,
            steps=[
                _step(1, '36 and 37', 'correct', 2.0),
            ],
            scheme=[
                _scheme(1, '36 and 37', 2.0, 'Complete the algebra accurately and state the final result'),
            ],
            feedback=[
                _fb(1, 'You correctly translated the words into algebra and solved for the requested quantities.',
                    improve='Define the unknown clearly, write an equation from every condition, solve it, and check the result in the original wording.'),
            ],
            topic='algebraic_word_problems', difficulty='easy',
        ),
        _example(
            'A rectangle has length 4 cm more than its width and perimeter 40 cm. Find its dimensions.', 'algebraic modelling', 1.0, 2.0,
            steps=[
                _step(1, '2(w + w + 4) = 40', 'partial', 1.0, 'The perimeter equation is correct, but the value of w was not solved.'),
            ],
            scheme=[
                _scheme(1, 'width = 8 cm, length = 12 cm', 2.0, 'Complete the algebra accurately and state the final result'),
            ],
            feedback=[
                _fb(1, 'You made a valid start and used part of the correct algebraic method.',
                    missing='The perimeter equation is correct, but the value of w was not solved. The complete result is width = 8 cm, length = 12 cm.',
                    deduction='1 mark deducted because the response is incomplete or not fully simplified.',
                    improve='Define the unknown clearly, write an equation from every condition, solve it, and check the result in the original wording.'),
            ],
            topic='algebraic_word_problems', difficulty='medium',
        ),
        _example(
            'After a 25% increase, a price becomes 90. Find the original price.', 'algebraic modelling', 1.0, 2.0,
            steps=[
                _step(1, '1.25x = 90', 'partial', 1.0, 'The percentage equation is correct, but x was not isolated.'),
            ],
            scheme=[
                _scheme(1, '72', 2.0, 'Complete the algebra accurately and state the final result'),
            ],
            feedback=[
                _fb(1, 'You made a valid start and used part of the correct algebraic method.',
                    missing='The percentage equation is correct, but x was not isolated. The complete result is 72.',
                    deduction='1 mark deducted because the response is incomplete or not fully simplified.',
                    improve='Define the unknown clearly, write an equation from every condition, solve it, and check the result in the original wording.'),
            ],
            topic='algebraic_word_problems', difficulty='difficult',
        ),
        _example(
            'Three consecutive even integers have sum 126. Find them.', 'algebraic modelling', 0.0, 2.0,
            steps=[
                _step(1, '41, 42, and 43', 'incorrect', 0.0, 'Consecutive integers were used instead of consecutive even integers.'),
            ],
            scheme=[
                _scheme(1, '40, 42, and 44', 2.0, 'Complete the algebra accurately and state the final result'),
            ],
            feedback=[
                _fb(1, 'You identified the relevant algebra topic and attempted a solution.',
                    missing='Consecutive integers were used instead of consecutive even integers. The correct result is 40, 42, and 44.',
                    deduction='Full marks lost because the stated result does not follow the required algebraic rule.',
                    improve='Define the unknown clearly, write an equation from every condition, solve it, and check the result in the original wording.'),
            ],
            topic='algebraic_word_problems', difficulty='medium',
        ),
        _example(
            'A taxi charges a fixed fee of 6 plus 3 per kilometre. A journey costs 39. Find the distance.', 'algebraic modelling', 0.0, 2.0,
            steps=[
                _step(1, '13 km', 'incorrect', 0.0, 'The fixed fee was not subtracted before dividing by the per-kilometre rate.'),
            ],
            scheme=[
                _scheme(1, '11 km', 2.0, 'Complete the algebra accurately and state the final result'),
            ],
            feedback=[
                _fb(1, 'You identified the relevant algebra topic and attempted a solution.',
                    missing='The fixed fee was not subtracted before dividing by the per-kilometre rate. The correct result is 11 km.',
                    deduction='Full marks lost because the stated result does not follow the required algebraic rule.',
                    improve='Define the unknown clearly, write an equation from every condition, solve it, and check the result in the original wording.'),
            ],
            topic='algebraic_word_problems', difficulty='difficult',
        ),
        _example(
            'Divide 42 in the ratio 2:5', 'ratio and proportion', 2.0, 2.0,
            steps=[
                _step(1, '12 and 30', 'correct', 2.0),
            ],
            scheme=[
                _scheme(1, '12 and 30', 2.0, 'Complete the algebra accurately and state the final result'),
            ],
            feedback=[
                _fb(1, 'You correctly set up the proportional relationship and calculated the required values.',
                    improve='Identify the total number of parts or the constant of proportionality before calculating the final values.'),
            ],
            topic='ratio_proportion', difficulty='easy',
        ),
        _example(
            'y is directly proportional to x. When x=3, y=9. Find y when x=8.', 'ratio and proportion', 1.0, 2.0,
            steps=[
                _step(1, 'y = kx and k = 3', 'partial', 1.0, 'The constant of proportionality was found, but it was not used with the new x value.'),
            ],
            scheme=[
                _scheme(1, '24', 2.0, 'Complete the algebra accurately and state the final result'),
            ],
            feedback=[
                _fb(1, 'You made a valid start and used part of the correct algebraic method.',
                    missing='The constant of proportionality was found, but it was not used with the new x value. The complete result is 24.',
                    deduction='1 mark deducted because the response is incomplete or not fully simplified.',
                    improve='Identify the total number of parts or the constant of proportionality before calculating the final values.'),
            ],
            topic='ratio_proportion', difficulty='medium',
        ),
        _example(
            'If a:b = 4:5 and b:c = 5:7, find a:b:c', 'ratio and proportion', 1.0, 2.0,
            steps=[
                _step(1, 'The common b value is 5', 'partial', 1.0, 'The common middle term was identified, but the three-part ratio was not written.'),
            ],
            scheme=[
                _scheme(1, '4:5:7', 2.0, 'Complete the algebra accurately and state the final result'),
            ],
            feedback=[
                _fb(1, 'You made a valid start and used part of the correct algebraic method.',
                    missing='The common middle term was identified, but the three-part ratio was not written. The complete result is 4:5:7.',
                    deduction='1 mark deducted because the response is incomplete or not fully simplified.',
                    improve='Identify the total number of parts or the constant of proportionality before calculating the final values.'),
            ],
            topic='ratio_proportion', difficulty='difficult',
        ),
        _example(
            'Divide 117 in the ratio 5:8', 'ratio and proportion', 0.0, 2.0,
            steps=[
                _step(1, '23 and 14', 'incorrect', 0.0, 'The total was divided separately by each ratio number instead of by the sum of the parts.'),
            ],
            scheme=[
                _scheme(1, '45 and 72', 2.0, 'Complete the algebra accurately and state the final result'),
            ],
            feedback=[
                _fb(1, 'You identified the relevant algebra topic and attempted a solution.',
                    missing='The total was divided separately by each ratio number instead of by the sum of the parts. The correct result is 45 and 72.',
                    deduction='Full marks lost because the stated result does not follow the required algebraic rule.',
                    improve='Identify the total number of parts or the constant of proportionality before calculating the final values.'),
            ],
            topic='ratio_proportion', difficulty='medium',
        ),
        _example(
            'y is directly proportional to x. When x=6, y=18. Find y when x=11.', 'ratio and proportion', 0.0, 2.0,
            steps=[
                _step(1, '23', 'incorrect', 0.0, 'Direct proportion requires multiplication by the constant k, not adding the change in x.'),
            ],
            scheme=[
                _scheme(1, '33', 2.0, 'Complete the algebra accurately and state the final result'),
            ],
            feedback=[
                _fb(1, 'You identified the relevant algebra topic and attempted a solution.',
                    missing='Direct proportion requires multiplication by the constant k, not adding the change in x. The correct result is 33.',
                    deduction='Full marks lost because the stated result does not follow the required algebraic rule.',
                    improve='Identify the total number of parts or the constant of proportionality before calculating the final values.'),
            ],
            topic='ratio_proportion', difficulty='difficult',
        ),
        _example(
            'Make x the subject of y = 2x - 3', 'rearranging formulas', 2.0, 2.0,
            steps=[
                _step(1, 'x = (y + 3)/2', 'correct', 2.0),
            ],
            scheme=[
                _scheme(1, 'x = (y + 3)/2', 2.0, 'Complete the algebra accurately and state the final result'),
            ],
            feedback=[
                _fb(1, 'You correctly used inverse operations and isolated x completely.',
                    improve='Apply the same inverse operation to both sides, collect x terms, and divide only after x is isolated as a factor.'),
            ],
            topic='rearranging_formulas', difficulty='easy',
        ),
        _example(
            'Make x the subject of y = (3x + 4)/4', 'rearranging formulas', 1.0, 2.0,
            steps=[
                _step(1, '(4)y = 3x + 4', 'partial', 1.0, 'The equation was rearranged correctly to one step before isolating x, but the final division was not done.'),
            ],
            scheme=[
                _scheme(1, 'x = ((4)y - 4)/3', 2.0, 'Complete the algebra accurately and state the final result'),
            ],
            feedback=[
                _fb(1, 'You made a valid start and used part of the correct algebraic method.',
                    missing='The equation was rearranged correctly to one step before isolating x, but the final division was not done. The complete result is x = ((4)y - 4)/3.',
                    deduction='1 mark deducted because the response is incomplete or not fully simplified.',
                    improve='Apply the same inverse operation to both sides, collect x terms, and divide only after x is isolated as a factor.'),
            ],
            topic='rearranging_formulas', difficulty='medium',
        ),
        _example(
            'Make x the subject of y = 4x - 5', 'rearranging formulas', 1.0, 2.0,
            steps=[
                _step(1, 'y + 5 = 4x', 'partial', 1.0, 'The equation was rearranged correctly to one step before isolating x, but the final division was not done.'),
            ],
            scheme=[
                _scheme(1, 'x = (y + 5)/4', 2.0, 'Complete the algebra accurately and state the final result'),
            ],
            feedback=[
                _fb(1, 'You made a valid start and used part of the correct algebraic method.',
                    missing='The equation was rearranged correctly to one step before isolating x, but the final division was not done. The complete result is x = (y + 5)/4.',
                    deduction='1 mark deducted because the response is incomplete or not fully simplified.',
                    improve='Apply the same inverse operation to both sides, collect x terms, and divide only after x is isolated as a factor.'),
            ],
            topic='rearranging_formulas', difficulty='difficult',
        ),
        _example(
            'Make x the subject of y = (5x + 6)/6', 'rearranging formulas', 0.0, 2.0,
            steps=[
                _step(1, 'x = ((6)y + 6)/5', 'incorrect', 0.0, 'After clearing the denominator, +b must be subtracted before dividing by the x coefficient.'),
            ],
            scheme=[
                _scheme(1, 'x = ((6)y - 6)/5', 2.0, 'Complete the algebra accurately and state the final result'),
            ],
            feedback=[
                _fb(1, 'You identified the relevant algebra topic and attempted a solution.',
                    missing='After clearing the denominator, +b must be subtracted before dividing by the x coefficient. The correct result is x = ((6)y - 6)/5.',
                    deduction='Full marks lost because the stated result does not follow the required algebraic rule.',
                    improve='Apply the same inverse operation to both sides, collect x terms, and divide only after x is isolated as a factor.'),
            ],
            topic='rearranging_formulas', difficulty='medium',
        ),
        _example(
            'Make x the subject of y = 6x - 7', 'rearranging formulas', 0.0, 2.0,
            steps=[
                _step(1, 'x = (y - 7)/6', 'incorrect', 0.0, 'Moving -b to the other side requires adding b, not subtracting it again.'),
            ],
            scheme=[
                _scheme(1, 'x = (y + 7)/6', 2.0, 'Complete the algebra accurately and state the final result'),
            ],
            feedback=[
                _fb(1, 'You identified the relevant algebra topic and attempted a solution.',
                    missing='Moving -b to the other side requires adding b, not subtracting it again. The correct result is x = (y + 7)/6.',
                    deduction='Full marks lost because the stated result does not follow the required algebraic rule.',
                    improve='Apply the same inverse operation to both sides, collect x terms, and divide only after x is isolated as a factor.'),
            ],
            topic='rearranging_formulas', difficulty='difficult',
        ),
        _example(
            'Solve |x + 2| = 3', 'absolute value equation', 2.0, 2.0,
            steps=[
                _step(1, 'x = 1 or x = -5', 'correct', 2.0),
            ],
            scheme=[
                _scheme(1, 'x = 1 or x = -5', 2.0, 'Complete the algebra accurately and state the final result'),
            ],
            feedback=[
                _fb(1, 'You correctly split the absolute-value equation into positive and negative cases.',
                    improve='Always write and solve both equations inside=k and inside=-k, then verify both solutions.'),
            ],
            topic='absolute_value_equations', difficulty='easy',
        ),
        _example(
            'Solve |2x + 3| = 8', 'absolute value equation', 1.0, 2.0,
            steps=[
                _step(1, '2x + 3 = 8, so x = 5/2', 'partial', 1.0, 'The positive case was solved correctly, but the negative case was omitted.'),
            ],
            scheme=[
                _scheme(1, 'x = 5/2 or x = -11/2', 2.0, 'Complete the algebra accurately and state the final result'),
            ],
            feedback=[
                _fb(1, 'You made a valid start and used part of the correct algebraic method.',
                    missing='The positive case was solved correctly, but the negative case was omitted. The complete result is x = 5/2 or x = -11/2.',
                    deduction='1 mark deducted because the response is incomplete or not fully simplified.',
                    improve='Always write and solve both equations inside=k and inside=-k, then verify both solutions.'),
            ],
            topic='absolute_value_equations', difficulty='medium',
        ),
        _example(
            'Solve |3x + 4| = 15', 'absolute value equation', 1.0, 2.0,
            steps=[
                _step(1, '3x + 4 = 15, so x = 11/3', 'partial', 1.0, 'The positive case was solved correctly, but the negative case was omitted.'),
            ],
            scheme=[
                _scheme(1, 'x = 11/3 or x = -19/3', 2.0, 'Complete the algebra accurately and state the final result'),
            ],
            feedback=[
                _fb(1, 'You made a valid start and used part of the correct algebraic method.',
                    missing='The positive case was solved correctly, but the negative case was omitted. The complete result is x = 11/3 or x = -19/3.',
                    deduction='1 mark deducted because the response is incomplete or not fully simplified.',
                    improve='Always write and solve both equations inside=k and inside=-k, then verify both solutions.'),
            ],
            topic='absolute_value_equations', difficulty='difficult',
        ),
        _example(
            'Solve |4x + 5| = 24', 'absolute value equation', 0.0, 2.0,
            steps=[
                _step(1, 'x = 19/4', 'incorrect', 0.0, 'An equation |A|=k with k>0 requires both A=k and A=-k; only one case was considered.'),
            ],
            scheme=[
                _scheme(1, 'x = 19/4 or x = -29/4', 2.0, 'Complete the algebra accurately and state the final result'),
            ],
            feedback=[
                _fb(1, 'You identified the relevant algebra topic and attempted a solution.',
                    missing='An equation |A|=k with k>0 requires both A=k and A=-k; only one case was considered. The correct result is x = 19/4 or x = -29/4.',
                    deduction='Full marks lost because the stated result does not follow the required algebraic rule.',
                    improve='Always write and solve both equations inside=k and inside=-k, then verify both solutions.'),
            ],
            topic='absolute_value_equations', difficulty='medium',
        ),
        _example(
            'Solve |5x + 6| = 35', 'absolute value equation', 0.0, 2.0,
            steps=[
                _step(1, 'x = 29/5', 'incorrect', 0.0, 'An equation |A|=k with k>0 requires both A=k and A=-k; only one case was considered.'),
            ],
            scheme=[
                _scheme(1, 'x = 29/5 or x = -41/5', 2.0, 'Complete the algebra accurately and state the final result'),
            ],
            feedback=[
                _fb(1, 'You identified the relevant algebra topic and attempted a solution.',
                    missing='An equation |A|=k with k>0 requires both A=k and A=-k; only one case was considered. The correct result is x = 29/5 or x = -41/5.',
                    deduction='Full marks lost because the stated result does not follow the required algebraic rule.',
                    improve='Always write and solve both equations inside=k and inside=-k, then verify both solutions.'),
            ],
            topic='absolute_value_equations', difficulty='difficult',
        ),
        _example(
            'Solve 2^(x + 1) = 32', 'exponential equation with equal bases', 2.0, 2.0,
            steps=[
                _step(1, 'x = 4', 'correct', 2.0),
            ],
            scheme=[
                _scheme(1, 'x = 4', 2.0, 'Complete the algebra accurately and state the final result'),
            ],
            feedback=[
                _fb(1, 'You correctly rewrote the right side with the same base, equated exponents, and solved for x.',
                    improve='Rewrite the right side as a power of the same base, equate exponents, then solve the linear equation.'),
            ],
            topic='exponent_equations', difficulty='easy',
        ),
        _example(
            'Solve 3^(x + 2) = 2187', 'exponential equation with equal bases', 1.0, 2.0,
            steps=[
                _step(1, 'x + 2 = 7', 'partial', 1.0, 'The exponents were equated correctly, but the resulting linear equation was not solved.'),
            ],
            scheme=[
                _scheme(1, 'x = 5', 2.0, 'Complete the algebra accurately and state the final result'),
            ],
            feedback=[
                _fb(1, 'You made a valid start and used part of the correct algebraic method.',
                    missing='The exponents were equated correctly, but the resulting linear equation was not solved. The complete result is x = 5.',
                    deduction='1 mark deducted because the response is incomplete or not fully simplified.',
                    improve='Rewrite the right side as a power of the same base, equate exponents, then solve the linear equation.'),
            ],
            topic='exponent_equations', difficulty='medium',
        ),
        _example(
            'Solve 4^(x + 3) = 262144', 'exponential equation with equal bases', 1.0, 2.0,
            steps=[
                _step(1, 'x + 3 = 9', 'partial', 1.0, 'The exponents were equated correctly, but the resulting linear equation was not solved.'),
            ],
            scheme=[
                _scheme(1, 'x = 6', 2.0, 'Complete the algebra accurately and state the final result'),
            ],
            feedback=[
                _fb(1, 'You made a valid start and used part of the correct algebraic method.',
                    missing='The exponents were equated correctly, but the resulting linear equation was not solved. The complete result is x = 6.',
                    deduction='1 mark deducted because the response is incomplete or not fully simplified.',
                    improve='Rewrite the right side as a power of the same base, equate exponents, then solve the linear equation.'),
            ],
            topic='exponent_equations', difficulty='difficult',
        ),
        _example(
            'Solve 2^(x + 4) = 2048', 'exponential equation with equal bases', 0.0, 2.0,
            steps=[
                _step(1, 'x = 2044', 'incorrect', 0.0, 'Once both sides have the same base, equate the exponents, not the exponent with the large numerical value.'),
            ],
            scheme=[
                _scheme(1, 'x = 7', 2.0, 'Complete the algebra accurately and state the final result'),
            ],
            feedback=[
                _fb(1, 'You identified the relevant algebra topic and attempted a solution.',
                    missing='Once both sides have the same base, equate the exponents, not the exponent with the large numerical value. The correct result is x = 7.',
                    deduction='Full marks lost because the stated result does not follow the required algebraic rule.',
                    improve='Rewrite the right side as a power of the same base, equate exponents, then solve the linear equation.'),
            ],
            topic='exponent_equations', difficulty='medium',
        ),
        _example(
            'Solve 3^(x + 5) = 1594323', 'exponential equation with equal bases', 0.0, 2.0,
            steps=[
                _step(1, 'x = 1594318', 'incorrect', 0.0, 'Once both sides have the same base, equate the exponents, not the exponent with the large numerical value.'),
            ],
            scheme=[
                _scheme(1, 'x = 8', 2.0, 'Complete the algebra accurately and state the final result'),
            ],
            feedback=[
                _fb(1, 'You identified the relevant algebra topic and attempted a solution.',
                    missing='Once both sides have the same base, equate the exponents, not the exponent with the large numerical value. The correct result is x = 8.',
                    deduction='Full marks lost because the stated result does not follow the required algebraic rule.',
                    improve='Rewrite the right side as a power of the same base, equate exponents, then solve the linear equation.'),
            ],
            topic='exponent_equations', difficulty='difficult',
        ),
        _example(
            'Simplify 10x + 13 - 7x + 19', 'simplifying expressions', 2.0, 2.0,
            steps=[
                _step(1, '3x + 32', 'correct', 2.0),
            ],
            scheme=[
                _scheme(1, '3x + 32', 2.0, 'Complete the algebra accurately and state the final result'),
            ],
            feedback=[
                _fb(1, 'You correctly combined the x terms and the constant terms.',
                    improve='Group like terms first: combine x terms together and constants together.'),
            ],
            topic='simplifying_expressions', difficulty='easy',
        ),
        _example(
            'Collect like terms: 9x^2 + 15x - 6x^2 + 8x - 11', 'collecting like terms', 1.0, 2.0,
            steps=[
                _step(1, '3x^2 + 23x', 'partial', 1.0, 'The constant term was omitted from the final expression.'),
            ],
            scheme=[
                _scheme(1, '3x^2 + 23x - 11', 2.0, 'Complete the algebra accurately and state the final result'),
            ],
            feedback=[
                _fb(1, 'You made a valid start and used part of the correct algebraic method.',
                    missing='The constant term was omitted from the final expression. The complete result is 3x^2 + 23x - 11.',
                    deduction='1 mark deducted because the response is incomplete or not fully simplified.',
                    improve='Only combine terms whose variables and exponents match exactly, and retain unmatched terms.'),
            ],
            topic='collecting_like_terms', difficulty='easy',
        ),
        _example(
            'Expand and simplify (x + 7)(x - 10)', 'expanding brackets', 1.0, 2.0,
            steps=[
                _step(1, 'x^2 - 10x + 7x - 70', 'partial', 1.0, 'All four products are present, but the two x terms were not collected.'),
            ],
            scheme=[
                _scheme(1, 'x^2 - 3x - 70', 2.0, 'Complete the algebra accurately and state the final result'),
            ],
            feedback=[
                _fb(1, 'You made a valid start and used part of the correct algebraic method.',
                    missing='All four products are present, but the two x terms were not collected. The complete result is x^2 - 3x - 70.',
                    deduction='1 mark deducted because the response is incomplete or not fully simplified.',
                    improve='Use FOIL or full distribution, then collect the two middle x terms.'),
            ],
            topic='expanding_brackets', difficulty='easy',
        ),
        _example(
            'Factorise fully 56x^2 + 105x', 'highest common factor', 1.0, 2.0,
            steps=[
                _step(1, '7(8x^2 + 15x)', 'partial', 1.0, 'The numerical factor was removed, but the common factor x was not extracted.'),
            ],
            scheme=[
                _scheme(1, '7x(8x + 15)', 2.0, 'Complete the algebra accurately and state the final result'),
            ],
            feedback=[
                _fb(1, 'You made a valid start and used part of the correct algebraic method.',
                    missing='The numerical factor was removed, but the common factor x was not extracted. The complete result is 7x(8x + 15).',
                    deduction='1 mark deducted because the response is incomplete or not fully simplified.',
                    improve='Find the greatest common factor of both coefficients and variables, then divide each term by it.'),
            ],
            topic='factorising_common_factor', difficulty='easy',
        ),
        _example(
            'Factorise x^2 + 16x + 63', 'quadratic factorisation', 1.0, 2.0,
            steps=[
                _step(1, 'The required factor pair is 7 and 9', 'partial', 1.0, 'The correct factor pair was found, but it was not written as a factorised expression.'),
            ],
            scheme=[
                _scheme(1, '(x + 7)(x + 9)', 2.0, 'Complete the algebra accurately and state the final result'),
            ],
            feedback=[
                _fb(1, 'You made a valid start and used part of the correct algebraic method.',
                    missing='The correct factor pair was found, but it was not written as a factorised expression. The complete result is (x + 7)(x + 9).',
                    deduction='1 mark deducted because the response is incomplete or not fully simplified.',
                    improve='Check both conditions: the factors must multiply to the constant and add to the middle coefficient.'),
            ],
            topic='factorising_quadratic', difficulty='easy',
        ),
        _example(
            'Factorise 49x^2 - 64', 'difference of squares', 1.0, 2.0,
            steps=[
                _step(1, '(7x - 8)(...)', 'partial', 1.0, 'Only one of the two conjugate factors was completed.'),
            ],
            scheme=[
                _scheme(1, '(7x - 8)(7x + 8)', 2.0, 'Complete the algebra accurately and state the final result'),
            ],
            feedback=[
                _fb(1, 'You made a valid start and used part of the correct algebraic method.',
                    missing='Only one of the two conjugate factors was completed. The complete result is (7x - 8)(7x + 8).',
                    deduction='1 mark deducted because the response is incomplete or not fully simplified.',
                    improve='Identify A and B, then write one factor with a minus sign and the other with a plus sign.'),
            ],
            topic='difference_of_squares', difficulty='easy',
        ),
        _example(
            'Factorise 216x^3 + 343', 'sum-of-cubes identity', 1.0, 2.0,
            steps=[
                _step(1, '(6x + 7)(36x^2 + ... + 49)', 'partial', 1.0, 'The outer factor and square terms are correct, but the middle ab term is missing.'),
            ],
            scheme=[
                _scheme(1, '(6x + 7)(36x^2 - 42x + 49)', 2.0, 'Complete the algebra accurately and state the final result'),
            ],
            feedback=[
                _fb(1, 'You made a valid start and used part of the correct algebraic method.',
                    missing='The outer factor and square terms are correct, but the middle ab term is missing. The complete result is (6x + 7)(36x^2 - 42x + 49).',
                    deduction='1 mark deducted because the response is incomplete or not fully simplified.',
                    improve='Write the complete identity first, substitute A and B, and check the middle product AB carefully.'),
            ],
            topic='sum_difference_of_cubes', difficulty='easy',
        ),
        _example(
            'Solve 8x + 15 = 87', 'linear equation', 1.0, 2.0,
            steps=[
                _step(1, '8x = 72', 'partial', 1.0, 'The x term was isolated correctly, but the final division was not completed.'),
            ],
            scheme=[
                _scheme(1, 'x = 9', 2.0, 'Complete the algebra accurately and state the final result'),
            ],
            feedback=[
                _fb(1, 'You made a valid start and used part of the correct algebraic method.',
                    missing='The x term was isolated correctly, but the final division was not completed. The complete result is x = 9.',
                    deduction='1 mark deducted because the response is incomplete or not fully simplified.',
                    improve='Undo addition or subtraction first, then divide both sides by the coefficient of x.'),
            ],
            topic='linear_equations', difficulty='easy',
        ),
        _example(
            'Solve 7(x + 6) = 98', 'linear equation with brackets', 1.0, 2.0,
            steps=[
                _step(1, '7x = 56', 'partial', 1.0, 'The bracket was expanded and x was isolated, but division by the coefficient was not completed.'),
            ],
            scheme=[
                _scheme(1, 'x = 8', 2.0, 'Complete the algebra accurately and state the final result'),
            ],
            feedback=[
                _fb(1, 'You made a valid start and used part of the correct algebraic method.',
                    missing='The bracket was expanded and x was isolated, but division by the coefficient was not completed. The complete result is x = 8.',
                    deduction='1 mark deducted because the response is incomplete or not fully simplified.',
                    improve='Distribute to every term, isolate the x term, and complete the final division.'),
            ],
            topic='equations_with_brackets', difficulty='easy',
        ),
        _example(
            'Solve (x + 7)/8 = 9', 'linear equation with fractions', 1.0, 2.0,
            steps=[
                _step(1, 'x + 7 = 72', 'partial', 1.0, 'The fraction was cleared correctly, but the constant was not moved to finish solving for x.'),
            ],
            scheme=[
                _scheme(1, 'x = 65', 2.0, 'Complete the algebra accurately and state the final result'),
            ],
            feedback=[
                _fb(1, 'You made a valid start and used part of the correct algebraic method.',
                    missing='The fraction was cleared correctly, but the constant was not moved to finish solving for x. The complete result is x = 65.',
                    deduction='1 mark deducted because the response is incomplete or not fully simplified.',
                    improve='Multiply both sides by the denominator, then use the inverse operation to isolate x.'),
            ],
            topic='equations_with_fractions', difficulty='easy',
        ),
        _example(
            'Solve -7x + 9 < -40', 'linear inequality', 1.0, 2.0,
            steps=[
                _step(1, '-7x < -49', 'partial', 1.0, 'The x term was isolated, but the final division and inequality direction were not completed.'),
            ],
            scheme=[
                _scheme(1, 'x > 7', 2.0, 'Complete the algebra accurately and state the final result'),
            ],
            feedback=[
                _fb(1, 'You made a valid start and used part of the correct algebraic method.',
                    missing='The x term was isolated, but the final division and inequality direction were not completed. The complete result is x > 7.',
                    deduction='1 mark deducted because the response is incomplete or not fully simplified.',
                    improve='Isolate the variable and remember to reverse the sign only when multiplying or dividing by a negative value.'),
            ],
            topic='linear_inequalities', difficulty='easy',
        ),
        _example(
            'Solve x^2 - 15x + 54 < 0', 'quadratic inequality', 1.0, 2.0,
            steps=[
                _step(1, 'x < 9', 'partial', 1.0, 'Only one boundary or interval was stated, so the full solution set is incomplete.'),
            ],
            scheme=[
                _scheme(1, '6 < x < 9', 2.0, 'Complete the algebra accurately and state the final result'),
            ],
            feedback=[
                _fb(1, 'You made a valid start and used part of the correct algebraic method.',
                    missing='Only one boundary or interval was stated, so the full solution set is incomplete. The complete result is 6 < x < 9.',
                    deduction='1 mark deducted because the response is incomplete or not fully simplified.',
                    improve='Factorise, mark both roots on a sign diagram, and test the intervals before stating the solution set.'),
            ],
            topic='quadratic_inequalities', difficulty='easy',
        ),
        _example(
            'Solve y = 7x - 36, 8x + y = 69', 'substitution', 1.0, 2.0,
            steps=[
                _step(1, 'x = 7', 'partial', 1.0, 'The value of x is correct, but y was not found by substituting back.'),
            ],
            scheme=[
                _scheme(1, 'x = 7, y = 13', 2.0, 'Complete the algebra accurately and state the final result'),
            ],
            feedback=[
                _fb(1, 'You made a valid start and used part of the correct algebraic method.',
                    missing='The value of x is correct, but y was not found by substituting back. The complete result is x = 7, y = 13.',
                    deduction='1 mark deducted because the response is incomplete or not fully simplified.',
                    improve='After finding one variable, always substitute it into an original equation to calculate and verify the other.'),
            ],
            topic='simultaneous_substitution', difficulty='easy',
        ),
        _example(
            'Solve x + y = 15, x - y = 3', 'elimination', 1.0, 2.0,
            steps=[
                _step(1, '2x = 18, so x = 9', 'partial', 1.0, 'Elimination correctly produced x, but y was not calculated by substitution.'),
            ],
            scheme=[
                _scheme(1, 'x = 9, y = 6', 2.0, 'Complete the algebra accurately and state the final result'),
            ],
            feedback=[
                _fb(1, 'You made a valid start and used part of the correct algebraic method.',
                    missing='Elimination correctly produced x, but y was not calculated by substitution. The complete result is x = 9, y = 6.',
                    deduction='1 mark deducted because the response is incomplete or not fully simplified.',
                    improve='Use elimination for one variable, substitute back for the second, and check both equations.'),
            ],
            topic='simultaneous_elimination', difficulty='easy',
        ),
        _example(
            'Solve y = x + 8, y = x^2 - 34', 'substitution (linear-quadratic)', 1.0, 2.0,
            steps=[
                _step(1, '(7, 15)', 'partial', 1.0, 'Only one of the two intersection points was stated.'),
            ],
            scheme=[
                _scheme(1, '(7, 15) and (-6, 2)', 2.0, 'Complete the algebra accurately and state the final result'),
            ],
            feedback=[
                _fb(1, 'You made a valid start and used part of the correct algebraic method.',
                    missing='Only one of the two intersection points was stated. The complete result is (7, 15) and (-6, 2).',
                    deduction='1 mark deducted because the response is incomplete or not fully simplified.',
                    improve='A quadratic normally gives two roots; substitute each root into the linear equation to obtain both points.'),
            ],
            topic='simultaneous_linear_quadratic', difficulty='medium',
        ),
        _example(
            'Solve x^2 - 17x + 70 = 0 by factorisation', 'quadratic factorisation', 1.0, 2.0,
            steps=[
                _step(1, '(x - 7)(x - 10) = 0', 'partial', 1.0, 'The factorisation is correct, but the two roots were not stated.'),
            ],
            scheme=[
                _scheme(1, 'x = 7 or x = 10', 2.0, 'Complete the algebra accurately and state the final result'),
            ],
            feedback=[
                _fb(1, 'You made a valid start and used part of the correct algebraic method.',
                    missing='The factorisation is correct, but the two roots were not stated. The complete result is x = 7 or x = 10.',
                    deduction='1 mark deducted because the response is incomplete or not fully simplified.',
                    improve='After factorising, set each factor equal to zero and solve each resulting linear equation.'),
            ],
            topic='quadratic_factorisation', difficulty='medium',
        ),
        _example(
            'Solve x^2 + 2x - 48 = 0 using the quadratic formula', 'quadratic formula', 1.0, 2.0,
            steps=[
                _step(1, 'x = (-2 ± √(196))/2', 'partial', 1.0, 'The formula was set up correctly, but the square root and the two final cases were not evaluated.'),
            ],
            scheme=[
                _scheme(1, 'x = 6 or x = -8', 2.0, 'Complete the algebra accurately and state the final result'),
            ],
            feedback=[
                _fb(1, 'You made a valid start and used part of the correct algebraic method.',
                    missing='The formula was set up correctly, but the square root and the two final cases were not evaluated. The complete result is x = 6 or x = -8.',
                    deduction='1 mark deducted because the response is incomplete or not fully simplified.',
                    improve='Write a, b, and c with signs, substitute into -b±√(b²-4ac), and evaluate both cases separately.'),
            ],
            topic='quadratic_formula', difficulty='medium',
        ),
        _example(
            'Solve x^2 + 20x + 51 = 0 by completing the square', 'completing the square', 0.0, 2.0,
            steps=[
                _step(1, '(x + 20)^2 = 49', 'incorrect', 0.0, 'The number inside the completed square must be half the x coefficient, not the full coefficient.'),
            ],
            scheme=[
                _scheme(1, 'x = -3 or x = -17', 2.0, 'Complete the algebra accurately and state the final result'),
            ],
            feedback=[
                _fb(1, 'You identified the relevant algebra topic and attempted a solution.',
                    missing='The number inside the completed square must be half the x coefficient, not the full coefficient. The correct result is x = -3 or x = -17.',
                    deduction='Full marks lost because the stated result does not follow the required algebraic rule.',
                    improve='Use half the x coefficient inside the bracket, balance the added square, then use ± when taking square roots.'),
            ],
            topic='completing_the_square', difficulty='medium',
        ),
        _example(
            'Simplify (x^2 - 64)/(x - 8)', 'simplifying algebraic fractions', 0.0, 2.0,
            steps=[
                _step(1, 'x - 8', 'incorrect', 0.0, 'After factorising (x-a)(x+a), cancelling (x-a) leaves x+a, not x-a.'),
            ],
            scheme=[
                _scheme(1, 'x + 8, x ≠ 8', 2.0, 'Complete the algebra accurately and state the final result'),
            ],
            feedback=[
                _fb(1, 'You identified the relevant algebra topic and attempted a solution.',
                    missing='After factorising (x-a)(x+a), cancelling (x-a) leaves x+a, not x-a. The correct result is x + 8, x ≠ 8.',
                    deduction='Full marks lost because the stated result does not follow the required algebraic rule.',
                    improve='Factorise first, cancel only common factors, and retain restrictions from the original denominator.'),
            ],
            topic='algebraic_fractions', difficulty='medium',
        ),
        _example(
            'Add (7x^2 + 13x - 9) and (6x^2 - 10x + 12)', 'polynomial addition', 0.0, 2.0,
            steps=[
                _step(1, '13x^2 + 23x + 21', 'incorrect', 0.0, 'The negative signs in the second polynomial were ignored when combining the x and constant terms.'),
            ],
            scheme=[
                _scheme(1, '13x^2 + 3x + 3', 2.0, 'Complete the algebra accurately and state the final result'),
            ],
            feedback=[
                _fb(1, 'You identified the relevant algebra topic and attempted a solution.',
                    missing='The negative signs in the second polynomial were ignored when combining the x and constant terms. The correct result is 13x^2 + 3x + 3.',
                    deduction='Full marks lost because the stated result does not follow the required algebraic rule.',
                    improve='Align equal powers vertically, keep every sign, and then add their coefficients.'),
            ],
            topic='polynomial_addition_subtraction', difficulty='medium',
        ),
        _example(
            'Multiply (x + 7)(x^2 + 6x + 8)', 'polynomial multiplication', 0.0, 2.0,
            steps=[
                _step(1, 'x^3 + 13x^2 + 56', 'incorrect', 0.0, 'The x terms created by cross-products were omitted from the final polynomial.'),
            ],
            scheme=[
                _scheme(1, 'x^3 + 13x^2 + 50x + 56', 2.0, 'Complete the algebra accurately and state the final result'),
            ],
            feedback=[
                _fb(1, 'You identified the relevant algebra topic and attempted a solution.',
                    missing='The x terms created by cross-products were omitted from the final polynomial. The correct result is x^3 + 13x^2 + 50x + 56.',
                    deduction='Full marks lost because the stated result does not follow the required algebraic rule.',
                    improve='Distribute each term across the full polynomial, then collect x³, x², x, and constants separately.'),
            ],
            topic='polynomial_multiplication', difficulty='medium',
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
        + _simplifying_expressions_examples()
        + _collecting_like_terms_examples()
        + _factorising_quadratic_examples()
        + _polynomial_addition_subtraction_examples()
        + _polynomial_multiplication_examples()
        + _polynomial_division_examples()
        + _functions_examples()
        + _function_composition_examples()
        + _inverse_functions_examples()
        + _straight_line_examples()
        + _gradient_intercept_examples()
        + _sequences_nth_term_examples()
        + _algebraic_word_problems_examples()
        + _ratio_proportion_examples()
        + _rearranging_formulas_examples()
        + _absolute_value_examples()
        + _exponent_equations_examples()
        + _equations_with_brackets_topup_examples()
        + _equations_with_fractions_topup_examples()
        + _simultaneous_substitution_topup_examples()
        + _negative_fractional_indices_topup_examples()
        + _rationalising_denominators_topup_examples()
        + _expanding_brackets_topup_examples()
        + _difference_of_squares_topup_examples()
        + _sum_diff_cubes_deepen_examples()
        + _quadratic_formula_deepen_examples()
        + _completing_square_deepen_examples()
        + _surds_deepen_examples()
        + _linear_inequalities_deepen_examples()
        + _quadratic_inequalities_deepen_examples()
        + _simultaneous_elimination_deepen_examples()
        + _function_composition_deepen_examples()
        + _inverse_functions_deepen_examples()
        + _gradient_intercept_deepen_examples()
        + _sequences_nth_term_deepen_examples()
        + _ratio_proportion_deepen_examples()
        + _absolute_value_deepen_examples()
        + _exponent_equations_deepen_examples()
        + _validity_rebalance_examples()
        + _partial_credit_examples()
        + _broad_topic_pass_examples()
        + _bulk_generated_examples()
    )

    # Assign ids centrally (topic-scoped running counter) so uniqueness is
    # guaranteed mechanically rather than by hand-typed ids per example.
    counters: Dict[str, int] = {}
    for s in samples:
        counters[s["topic"]] = counters.get(s["topic"], 0) + 1
        s["id"] = f"{s['topic']}-{counters[s['topic']]:04d}"

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
    train_raw, eval_raw = split_raw_dataset(raw)
    train_set = [format_example(item) for item in train_raw]
    eval_set = [format_example(item) for item in eval_raw]
    _write_json(train_set, "app/training/data/feedback_dataset_train.json")
    _write_json(eval_set, "app/training/data/feedback_dataset_eval.json")
