"""
generate_al_cases.py
--------------------
Run this script from inside the `tests/` folder:

    cd services/reasoning-service/tests
    python generate_al_cases.py

It will create al_algebra_cases/ with 10 topic sub-folders and 30 JSON files.
"""

import json
import os
from pathlib import Path

BASE_DIR = Path(__file__).parent / "al_algebra_cases"

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def write(subfolder: str, filename: str, data: dict):
    folder = BASE_DIR / subfolder
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / filename
    data.pop("metadata", None)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"  ✔  {subfolder}/{filename}")


# ===========================================================================
# 1.  MATHEMATICAL INDUCTION  (q01 – q03)
# ===========================================================================

def q01_induction_perfect():
    return {
        "metadata": {"topic": "induction", "solution_type": "perfect", "question_number": 1},
        "reasoning_input": {
            "question_text": (
                "Prove by mathematical induction that for all positive integers n, "
                "1 + 2 + 3 + ... + n = n(n+1)/2."
            ),
            "student_steps": [
                {"step_id": 1,  "content": "Base case (n=1): LHS = 1. RHS = 1(1+1)/2 = 1. LHS = RHS. ✓"},
                {"step_id": 2,  "content": "Inductive hypothesis: Assume true for n = k, i.e., 1+2+...+k = k(k+1)/2."},
                {"step_id": 3,  "content": "Inductive step: Show true for n = k+1."},
                {"step_id": 4,  "content": "LHS for n=k+1: 1+2+...+k+(k+1)"},
                {"step_id": 5,  "content": "= k(k+1)/2 + (k+1)   [using hypothesis]"},
                {"step_id": 6,  "content": "= (k+1)[k/2 + 1]"},
                {"step_id": 7,  "content": "= (k+1)(k+2)/2"},
                {"step_id": 8,  "content": "= (k+1)((k+1)+1)/2, which is the formula with n = k+1."},
                {"step_id": 9,  "content": "Conclusion: By the principle of mathematical induction, the formula holds for all positive integers n."},
            ],
            "final_answer": "Proved by mathematical induction."
        },
        "marking_scheme": {
            "total_marks": 10,
            "steps": [
                {"step_no": 1, "description": "Verify base case n=1 correctly.", "marks": 1},
                {"step_no": 2, "description": "State inductive hypothesis for n=k clearly.", "marks": 2},
                {"step_no": 3, "description": "Set up the sum for n=k+1 using the hypothesis.", "marks": 2},
                {"step_no": 4, "description": "Algebraic manipulation to reach (k+1)(k+2)/2.", "marks": 3},
                {"step_no": 5, "description": "Recognise result matches formula for n=k+1.", "marks": 1},
                {"step_no": 6, "description": "State conclusion invoking the induction principle.", "marks": 1},
            ]
        }
    }


def q02_induction_missing_hypothesis():
    return {
        "metadata": {"topic": "induction", "solution_type": "partially_correct", "question_number": 2},
        "reasoning_input": {
            "question_text": (
                "Prove by mathematical induction that for all positive integers n, "
                "1² + 2² + 3² + ... + n² = n(n+1)(2n+1)/6."
            ),
            "student_steps": [
                {"step_id": 1,  "content": "Base case (n=1): LHS = 1. RHS = 1(2)(3)/6 = 1. True."},
                {"step_id": 2,  "content": "Assume the result is true for some value of n."},
                {"step_id": 3,  "content": "For n = k+1: LHS = 1²+2²+...+k²+(k+1)²"},
                {"step_id": 4,  "content": "= k(k+1)(2k+1)/6 + (k+1)²"},
                {"step_id": 5,  "content": "= (k+1)[k(2k+1)/6 + (k+1)]"},
                {"step_id": 6,  "content": "= (k+1)[2k²+k+6k+6]/6"},
                {"step_id": 7,  "content": "= (k+1)(2k²+7k+6)/6"},
                {"step_id": 8,  "content": "= (k+1)(k+2)(2k+3)/6"},
                {"step_id": 9,  "content": "This is the formula for n = k+1. Proved."},
            ],
            "final_answer": "Proved by induction."
        },
        "marking_scheme": {
            "total_marks": 10,
            "steps": [
                {"step_no": 1, "description": "Verify base case n=1.", "marks": 1},
                {"step_no": 2, "description": "State hypothesis explicitly for n=k (student vague — loses 1 mark).", "marks": 1},
                {"step_no": 3, "description": "Set up LHS for n=k+1.", "marks": 1},
                {"step_no": 4, "description": "Substitute k(k+1)(2k+1)/6 correctly.", "marks": 2},
                {"step_no": 5, "description": "Factor and simplify to (k+1)(k+2)(2k+3)/6.", "marks": 3},
                {"step_no": 6, "description": "Recognise form matches formula for n=k+1.", "marks": 1},
                {"step_no": 7, "description": "State induction conclusion (student did state it).", "marks": 1},
            ]
        }
    }


def q03_induction_wrong_base_case():
    return {
        "metadata": {"topic": "induction", "solution_type": "wrong_base_case", "question_number": 3},
        "reasoning_input": {
            "question_text": (
                "Prove by mathematical induction that for all positive integers n, "
                "n³ - n is divisible by 6."
            ),
            "student_steps": [
                {"step_id": 1,  "content": "Base case (n=1): 1³-1 = 0. Student writes: 0/6 = 1, which is wrong."},
                {"step_id": 2,  "content": "Assume 6 | (k³ - k) for some positive integer k."},
                {"step_id": 3,  "content": "(k+1)³ - (k+1) = k³ + 3k² + 3k + 1 - k - 1"},
                {"step_id": 4,  "content": "= k³ - k + 3k² + 3k"},
                {"step_id": 5,  "content": "= (k³ - k) + 3k(k+1)"},
                {"step_id": 6,  "content": "6 | (k³-k) by hypothesis."},
                {"step_id": 7,  "content": "k(k+1) is a product of consecutive integers, so 2 | k(k+1), hence 6 | 3k(k+1)."},
                {"step_id": 8,  "content": "Therefore 6 | [(k³-k) + 3k(k+1)] = (k+1)³-(k+1). Proved."},
            ],
            "final_answer": "Divisible by 6 for all positive integers n."
        },
        "marking_scheme": {
            "total_marks": 10,
            "steps": [
                {"step_no": 1, "description": "Correct base case: 1³-1=0, and 6|0 (student error here — loses 1 mark).", "marks": 0},
                {"step_no": 2, "description": "State inductive hypothesis for n=k clearly.", "marks": 2},
                {"step_no": 3, "description": "Expand (k+1)³-(k+1) correctly.", "marks": 2},
                {"step_no": 4, "description": "Regroup as (k³-k)+3k(k+1).", "marks": 2},
                {"step_no": 5, "description": "Argue divisibility by 6 of both parts.", "marks": 2},
                {"step_no": 6, "description": "State final conclusion.", "marks": 2},
            ]
        }
    }


# ===========================================================================
# 2.  PARTIAL FRACTIONS  (q04 – q06)
# ===========================================================================

def q04_partial_fractions_perfect():
    return {
        "metadata": {"topic": "partial_fractions", "solution_type": "perfect", "question_number": 4},
        "reasoning_input": {
            "question_text": (
                "Express (5x + 1) / [(x-1)(x+2)(x-3)] in partial fractions."
            ),
            "student_steps": [
                {"step_id": 1,  "content": "(5x+1)/[(x-1)(x+2)(x-3)] = A/(x-1) + B/(x+2) + C/(x-3)"},
                {"step_id": 2,  "content": "Multiply both sides by (x-1)(x+2)(x-3):"},
                {"step_id": 3,  "content": "5x+1 = A(x+2)(x-3) + B(x-1)(x-3) + C(x-1)(x+2)"},
                {"step_id": 4,  "content": "Let x=1: 6 = A(3)(-2) → A = -1"},
                {"step_id": 5,  "content": "Let x=-2: -9 = B(-3)(-5) = 15B → B = -3/5"},
                {"step_id": 6,  "content": "Let x=3: 16 = C(2)(5) = 10C → C = 8/5"},
                {"step_id": 7,  "content": "Result: -1/(x-1) + (-3/5)/(x+2) + (8/5)/(x-3)"},
                {"step_id": 8,  "content": "Verification: At x=0: LHS = 1/[(-1)(2)(-3)] = 1/6. RHS = -1/(-1) + (-3/5)/2 + (8/5)/(-3) = 1 - 3/10 - 8/15 = 1/6 ✓"},
            ],
            "final_answer": "-1/(x-1) - (3/5)/(x+2) + (8/5)/(x-3)"
        },
        "marking_scheme": {
            "total_marks": 9,
            "steps": [
                {"step_no": 1, "description": "Set up correct partial fraction form.", "marks": 1},
                {"step_no": 2, "description": "Multiply out and write identity correctly.", "marks": 1},
                {"step_no": 3, "description": "Find A by substituting x=1.", "marks": 2},
                {"step_no": 4, "description": "Find B by substituting x=-2.", "marks": 2},
                {"step_no": 5, "description": "Find C by substituting x=3.", "marks": 2},
                {"step_no": 6, "description": "State final answer clearly.", "marks": 1},
            ]
        }
    }


def q05_partial_fractions_partially_correct():
    return {
        "metadata": {"topic": "partial_fractions", "solution_type": "partially_correct", "question_number": 5},
        "reasoning_input": {
            "question_text": (
                "Express (3x² + 2x - 1) / [(x+1)(x²+4)] in partial fractions."
            ),
            "student_steps": [
                {"step_id": 1,  "content": "Student writes: A/(x+1) + B/(x²+4)   ← Wrong form for irreducible quadratic."},
                {"step_id": 2,  "content": "3x²+2x-1 = A(x²+4) + B(x+1)"},
                {"step_id": 3,  "content": "Let x=-1: 3-2-1 = A(1+4) → 0 = 5A → A = 0"},
                {"step_id": 4,  "content": "Compare x² coefficients: 3 = A + 0 → A = 3  [contradiction]"},
                {"step_id": 5,  "content": "Student is confused and cannot resolve."},
                {"step_id": 6,  "content": "Correct form should be: A/(x+1) + (Bx+C)/(x²+4)"},
            ],
            "final_answer": "Student could not complete — wrong partial fraction form used."
        },
        "marking_scheme": {
            "total_marks": 9,
            "steps": [
                {"step_no": 1, "description": "Correct form: A/(x+1) + (Bx+C)/(x²+4) — student failed here (0 marks).", "marks": 0},
                {"step_no": 2, "description": "Multiply and form identity.", "marks": 1},
                {"step_no": 3, "description": "Use x=-1 to find A = 0/5 — correct substitution attempted.", "marks": 1},
                {"step_no": 4, "description": "Compare coefficients to find B and C — not done.", "marks": 0},
                {"step_no": 5, "description": "State final answer.", "marks": 0},
            ]
        }
    }


def q06_partial_fractions_repeated_factor_perfect():
    return {
        "metadata": {"topic": "partial_fractions", "solution_type": "perfect_repeated_factor", "question_number": 6},
        "reasoning_input": {
            "question_text": (
                "Express (4x² - 3x + 5) / [(x-2)²(x+1)] in partial fractions."
            ),
            "student_steps": [
                {"step_id": 1,  "content": "Form: A/(x-2) + B/(x-2)² + C/(x+1)"},
                {"step_id": 2,  "content": "4x²-3x+5 = A(x-2)(x+1) + B(x+1) + C(x-2)²"},
                {"step_id": 3,  "content": "Let x=2: 4(4)-6+5=15. B(3)=15 → B=5"},
                {"step_id": 4,  "content": "Let x=-1: 4+3+5=12. C(-3)²=9C=12 → C=4/3"},
                {"step_id": 5,  "content": "Expand: A(x²-x-2)+5(x+1)+(4/3)(x²-4x+4)"},
                {"step_id": 6,  "content": "Compare x² coeff: 4 = A + 4/3 → A = 8/3"},
                {"step_id": 7,  "content": "Verify constant: A(-2(1)) + 5(1) + (4/3)(4) = -16/3 + 5 + 16/3 = 5 ✓"},
                {"step_id": 8,  "content": "Result: (8/3)/(x-2) + 5/(x-2)² + (4/3)/(x+1)"},
            ],
            "final_answer": "(8/3)/(x-2) + 5/(x-2)² + (4/3)/(x+1)"
        },
        "marking_scheme": {
            "total_marks": 10,
            "steps": [
                {"step_no": 1, "description": "Identify repeated factor form correctly.", "marks": 2},
                {"step_no": 2, "description": "Form identity by multiplying through.", "marks": 1},
                {"step_no": 3, "description": "Find B = 5 using x=2.", "marks": 2},
                {"step_no": 4, "description": "Find C = 4/3 using x=-1.", "marks": 2},
                {"step_no": 5, "description": "Find A = 8/3 by comparing coefficients.", "marks": 2},
                {"step_no": 6, "description": "State complete final answer.", "marks": 1},
            ]
        }
    }


# ===========================================================================
# 3.  BINOMIAL EXPANSION  (q07 – q09)
# ===========================================================================

def q07_binomial_perfect():
    return {
        "metadata": {"topic": "binomial", "solution_type": "perfect", "question_number": 7},
        "reasoning_input": {
            "question_text": (
                "(a) Expand (1 + 3x)^(-2) as a series in ascending powers of x up to and including the term in x³, "
                "stating the range of values of x for which the expansion is valid. "
                "(b) Use your expansion to find an approximation to 1/(1.03)² to 4 decimal places."
            ),
            "student_steps": [
                {"step_id": 1,  "content": "(1+3x)^(-2) = 1 + (-2)(3x) + (-2)(-3)/2! (3x)² + (-2)(-3)(-4)/3! (3x)³ + ..."},
                {"step_id": 2,  "content": "= 1 - 6x + (6)(9x²)/2 - (24)(27x³)/6 + ..."},
                {"step_id": 3,  "content": "= 1 - 6x + 27x² - 108x³ + ..."},
                {"step_id": 4,  "content": "Valid for |3x| < 1, i.e., |x| < 1/3."},
                {"step_id": 5,  "content": "Part (b): 1/(1.03)² = (1 + 3(0.01))^(-2), so x = 0.01."},
                {"step_id": 6,  "content": "≈ 1 - 6(0.01) + 27(0.0001) - 108(0.000001)"},
                {"step_id": 7,  "content": "= 1 - 0.06 + 0.0027 - 0.000108"},
                {"step_id": 8,  "content": "= 0.942592"},
            ],
            "final_answer": "(a) 1 - 6x + 27x² - 108x³, |x| < 1/3.  (b) ≈ 0.9426"
        },
        "marking_scheme": {
            "total_marks": 10,
            "steps": [
                {"step_no": 1, "description": "Correct use of binomial series formula for negative index.", "marks": 2},
                {"step_no": 2, "description": "Correct coefficients: 1, -6, 27, -108.", "marks": 3},
                {"step_no": 3, "description": "State validity |x| < 1/3.", "marks": 1},
                {"step_no": 4, "description": "Identify x = 0.01 correctly for approximation.", "marks": 1},
                {"step_no": 5, "description": "Substitute and compute each term correctly.", "marks": 2},
                {"step_no": 6, "description": "State final 4 d.p. answer 0.9426.", "marks": 1},
            ]
        }
    }


def q08_binomial_missing_validity():
    return {
        "metadata": {"topic": "binomial", "solution_type": "partially_correct_missing_validity", "question_number": 8},
        "reasoning_input": {
            "question_text": (
                "Expand (2 - x)^(-3) in ascending powers of x up to the term in x², "
                "stating the range of x for which the expansion is valid."
            ),
            "student_steps": [
                {"step_id": 1,  "content": "(2-x)^(-3) = 2^(-3)(1 - x/2)^(-3) = (1/8)(1 - x/2)^(-3)"},
                {"step_id": 2,  "content": "(1-x/2)^(-3) = 1 + (-3)(-x/2) + (-3)(-4)/2 (-x/2)² + ..."},
                {"step_id": 3,  "content": "= 1 + 3x/2 + 6(x²/4) + ..."},
                {"step_id": 4,  "content": "= 1 + 3x/2 + 3x²/2"},
                {"step_id": 5,  "content": "So (2-x)^(-3) = (1/8)(1 + 3x/2 + 3x²/2)"},
                {"step_id": 6,  "content": "= 1/8 + 3x/16 + 3x²/16"},
                {"step_id": 7,  "content": "Student does NOT state the range of validity."},
            ],
            "final_answer": "1/8 + 3x/16 + 3x²/16 (validity not stated)"
        },
        "marking_scheme": {
            "total_marks": 8,
            "steps": [
                {"step_no": 1, "description": "Factor out 2^(-3) = 1/8 correctly.", "marks": 1},
                {"step_no": 2, "description": "Apply binomial expansion correctly.", "marks": 2},
                {"step_no": 3, "description": "Simplify coefficients correctly.", "marks": 2},
                {"step_no": 4, "description": "Multiply by 1/8 to get final terms.", "marks": 2},
                {"step_no": 5, "description": "State validity |x| < 2 — NOT done. 0 marks.", "marks": 0},
            ]
        }
    }


def q09_binomial_approximation_wrong():
    return {
        "metadata": {"topic": "binomial", "solution_type": "wrong_approximation", "question_number": 9},
        "reasoning_input": {
            "question_text": (
                "Expand (1 + 2x)^(1/2) up to the term in x³. "
                "Hence find an approximation for √1.2 to 4 d.p."
            ),
            "student_steps": [
                {"step_id": 1,  "content": "(1+2x)^(1/2) = 1 + (1/2)(2x) + (1/2)(-1/2)/2 (2x)² + ..."},
                {"step_id": 2,  "content": "= 1 + x - x²/2 + ..."},
                {"step_id": 3,  "content": "Third term: (1/2)(-1/2)(-3/2)/6 (2x)³ = (3/8)(8x³)/6 = x³/2"},
                {"step_id": 4,  "content": "So (1+2x)^(1/2) ≈ 1 + x - x²/2 + x³/2"},
                {"step_id": 5,  "content": "For √1.2, student sets 2x = 0.2 → x = 0.1"},
                {"step_id": 6,  "content": "But student writes: √1.2 = 1 + 0.2 - (0.2)²/2 + (0.2)³/2   ← uses 0.2 instead of x=0.1"},
                {"step_id": 7,  "content": "= 1 + 0.2 - 0.02 + 0.004 = 1.184 (WRONG — x should be 0.1)"},
            ],
            "final_answer": "1.184 (incorrect — substitution error)"
        },
        "marking_scheme": {
            "total_marks": 10,
            "steps": [
                {"step_no": 1, "description": "Correct expansion terms up to x²: 1+x-x²/2.", "marks": 3},
                {"step_no": 2, "description": "Correct third term -x²/2 (note: correct sign).", "marks": 2},
                {"step_no": 3, "description": "Identify x = 0.1 (1+2(0.1) = 1.2). Student used x=0.2 — 0 marks.", "marks": 0},
                {"step_no": 4, "description": "Correct substitution and arithmetic — not done.", "marks": 0},
                {"step_no": 5, "description": "Final 4 d.p. answer 1.0954 — not reached.", "marks": 0},
            ]
        }
    }


# ===========================================================================
# 4.  LOGARITHMS  (q10 – q12)
# ===========================================================================

def q10_log_perfect():
    return {
        "metadata": {"topic": "logarithms", "solution_type": "perfect", "question_number": 10},
        "reasoning_input": {
            "question_text": (
                "Solve the equation log₂(x) + log₂(x - 6) = 4."
            ),
            "student_steps": [
                {"step_id": 1,  "content": "log₂[x(x-6)] = 4"},
                {"step_id": 2,  "content": "x(x-6) = 2⁴ = 16"},
                {"step_id": 3,  "content": "x² - 6x - 16 = 0"},
                {"step_id": 4,  "content": "(x-8)(x+2) = 0"},
                {"step_id": 5,  "content": "x = 8 or x = -2"},
                {"step_id": 6,  "content": "Check domain: log₂(x) requires x > 0 and log₂(x-6) requires x > 6."},
                {"step_id": 7,  "content": "x = -2 is rejected. x = 8: log₂(8) + log₂(2) = 3 + 1 = 4 ✓"},
            ],
            "final_answer": "x = 8"
        },
        "marking_scheme": {
            "total_marks": 8,
            "steps": [
                {"step_no": 1, "description": "Use product rule to combine logs.", "marks": 1},
                {"step_no": 2, "description": "Convert to exponential form: x(x-6) = 16.", "marks": 1},
                {"step_no": 3, "description": "Form and solve quadratic x²-6x-16=0.", "marks": 3},
                {"step_no": 4, "description": "Apply domain restriction and reject x=-2.", "marks": 2},
                {"step_no": 5, "description": "State x=8 with verification.", "marks": 1},
            ]
        }
    }


def q11_log_domain_error():
    return {
        "metadata": {"topic": "logarithms", "solution_type": "partially_correct_domain_error", "question_number": 11},
        "reasoning_input": {
            "question_text": (
                "Solve: log₅(2x+1) - log₅(x-1) = 1."
            ),
            "student_steps": [
                {"step_id": 1,  "content": "log₅[(2x+1)/(x-1)] = 1"},
                {"step_id": 2,  "content": "(2x+1)/(x-1) = 5"},
                {"step_id": 3,  "content": "2x+1 = 5(x-1) = 5x-5"},
                {"step_id": 4,  "content": "6 = 3x → x = 2"},
                {"step_id": 5,  "content": "Student states x = 2 without checking domain."},
            ],
            "final_answer": "x = 2 (domain check omitted)"
        },
        "marking_scheme": {
            "total_marks": 8,
            "steps": [
                {"step_no": 1, "description": "Use quotient rule correctly.", "marks": 1},
                {"step_no": 2, "description": "Convert to 5^1 = 5.", "marks": 1},
                {"step_no": 3, "description": "Solve linear equation → x = 2.", "marks": 3},
                {"step_no": 4, "description": "Verify x=2 satisfies domain (x>1, 2x+1>0) — not done. 0 marks.", "marks": 0},
                {"step_no": 5, "description": "State final answer.", "marks": 1},
            ]
        }
    }


def q12_log_change_base_partial():
    return {
        "metadata": {"topic": "logarithms", "solution_type": "partially_correct_change_base", "question_number": 12},
        "reasoning_input": {
            "question_text": (
                "Solve: log₄(x) = log₂(x) - 3."
            ),
            "student_steps": [
                {"step_id": 1,  "content": "log₄(x) = log₂(x) / log₂(4) = log₂(x) / 2"},
                {"step_id": 2,  "content": "Let y = log₂(x). Then y/2 = y - 3"},
                {"step_id": 3,  "content": "y = 2y - 6 → y = 6"},
                {"step_id": 4,  "content": "log₂(x) = 6 → x = 2⁶ = 64"},
                {"step_id": 5,  "content": "Student does not verify the answer."},
            ],
            "final_answer": "x = 64 (no verification)"
        },
        "marking_scheme": {
            "total_marks": 8,
            "steps": [
                {"step_no": 1, "description": "Change base: log₄(x) = log₂(x)/2 correctly.", "marks": 2},
                {"step_no": 2, "description": "Substitute y = log₂(x) and form equation.", "marks": 2},
                {"step_no": 3, "description": "Solve y = 6 correctly.", "marks": 2},
                {"step_no": 4, "description": "Convert to x = 64.", "marks": 1},
                {"step_no": 5, "description": "Verify: log₄(64)=3, log₂(64)=6, 6-3=3 ✓ — not done. 0 marks.", "marks": 0},
            ]
        }
    }


# ===========================================================================
# 5.  QUADRATIC  (q13 – q15)
# ===========================================================================

def q13_quadratic_completing_square_perfect():
    return {
        "metadata": {"topic": "quadratic", "solution_type": "perfect_completing_square", "question_number": 13},
        "reasoning_input": {
            "question_text": (
                "Express f(x) = 2x² - 8x + 11 in the form a(x+b)² + c. "
                "Hence state the minimum value of f(x) and the value of x at which it occurs."
            ),
            "student_steps": [
                {"step_id": 1,  "content": "f(x) = 2(x² - 4x) + 11"},
                {"step_id": 2,  "content": "= 2[(x-2)² - 4] + 11"},
                {"step_id": 3,  "content": "= 2(x-2)² - 8 + 11"},
                {"step_id": 4,  "content": "= 2(x-2)² + 3"},
                {"step_id": 5,  "content": "Minimum value = 3, occurs when (x-2)² = 0, i.e., x = 2."},
            ],
            "final_answer": "2(x-2)² + 3; minimum value 3 at x = 2."
        },
        "marking_scheme": {
            "total_marks": 7,
            "steps": [
                {"step_no": 1, "description": "Factor out 2 from x² term.", "marks": 1},
                {"step_no": 2, "description": "Complete the square inside brackets: (x-2)²-4.", "marks": 2},
                {"step_no": 3, "description": "Expand and simplify to 2(x-2)²+3.", "marks": 2},
                {"step_no": 4, "description": "State minimum value 3 and x=2.", "marks": 2},
            ]
        }
    }


def q14_quadratic_discriminant_sign_error():
    return {
        "metadata": {"topic": "quadratic", "solution_type": "partially_correct_discriminant_sign_error", "question_number": 14},
        "reasoning_input": {
            "question_text": (
                "Find the range of values of k for which the equation x² + kx + (k+3) = 0 has two distinct real roots."
            ),
            "student_steps": [
                {"step_id": 1,  "content": "For two distinct real roots: discriminant > 0"},
                {"step_id": 2,  "content": "b² - 4ac > 0: k² - 4(1)(k+3) > 0"},
                {"step_id": 3,  "content": "k² - 4k - 12 > 0"},
                {"step_id": 4,  "content": "(k-6)(k+2) > 0"},
                {"step_id": 5,  "content": "Student writes: -2 < k < 6  ← WRONG direction for quadratic inequality."},
            ],
            "final_answer": "-2 < k < 6 (incorrect — should be k < -2 or k > 6)"
        },
        "marking_scheme": {
            "total_marks": 7,
            "steps": [
                {"step_no": 1, "description": "State discriminant condition b²-4ac > 0.", "marks": 1},
                {"step_no": 2, "description": "Substitute correctly: k²-4k-12 > 0.", "marks": 2},
                {"step_no": 3, "description": "Factorise correctly: (k-6)(k+2) > 0.", "marks": 2},
                {"step_no": 4, "description": "Correct inequality direction: k<-2 or k>6 — student reversed it. 0 marks.", "marks": 0},
            ]
        }
    }


def q15_quadratic_formula_partial():
    return {
        "metadata": {"topic": "quadratic", "solution_type": "partially_correct_formula", "question_number": 15},
        "reasoning_input": {
            "question_text": (
                "Solve 3x² - 5x - 2 = 0 using the quadratic formula, giving exact answers."
            ),
            "student_steps": [
                {"step_id": 1,  "content": "a=3, b=-5, c=-2"},
                {"step_id": 2,  "content": "x = [5 ± √(25 + 24)] / 6"},
                {"step_id": 3,  "content": "x = [5 ± √49] / 6"},
                {"step_id": 4,  "content": "x = [5 ± 7] / 6"},
                {"step_id": 5,  "content": "x = 12/6 = 2 or x = -2/6"},
                {"step_id": 6,  "content": "Student writes x = -2/6 without simplifying to -1/3."},
            ],
            "final_answer": "x = 2 or x = -2/6 (should simplify to -1/3)"
        },
        "marking_scheme": {
            "total_marks": 6,
            "steps": [
                {"step_no": 1, "description": "Identify a, b, c correctly.", "marks": 1},
                {"step_no": 2, "description": "Apply formula correctly.", "marks": 2},
                {"step_no": 3, "description": "Evaluate discriminant √49 = 7.", "marks": 1},
                {"step_no": 4, "description": "Find x = 2 correctly.", "marks": 1},
                {"step_no": 5, "description": "Simplify -2/6 to -1/3 — not done. 0 marks.", "marks": 0},
            ]
        }
    }


# ===========================================================================
# 6.  POLYNOMIALS  (q16 – q18)
# ===========================================================================

def q16_polynomial_remainder_perfect():
    return {
        "metadata": {"topic": "polynomials", "solution_type": "perfect_remainder_theorem", "question_number": 16},
        "reasoning_input": {
            "question_text": (
                "The polynomial p(x) = 2x³ + ax² - bx + 3 leaves remainder 15 when divided by (x-2) "
                "and remainder -6 when divided by (x+1). Find a and b."
            ),
            "student_steps": [
                {"step_id": 1,  "content": "By Remainder Theorem: p(2) = 15 and p(-1) = -6."},
                {"step_id": 2,  "content": "p(2) = 2(8) + 4a - 2b + 3 = 16 + 4a - 2b + 3 = 19 + 4a - 2b = 15"},
                {"step_id": 3,  "content": "→ 4a - 2b = -4 → 2a - b = -2   ... (1)"},
                {"step_id": 4,  "content": "p(-1) = 2(-1) + a(-1)² - b(-1) + 3 = -2 + a + b + 3 = 1 + a + b = -6"},
                {"step_id": 5,  "content": "→ a + b = -7   ... (2)"},
                {"step_id": 6,  "content": "From (1): b = 2a+2. Substitute into (2): a + 2a+2 = -7 → 3a = -9 → a = -3"},
                {"step_id": 7,  "content": "b = 2(-3)+2 = -4"},
                {"step_id": 8,  "content": "Check: p(2) = 16 + 4(-3) -2(-4) +3 = 16-12+8+3 = 15 ✓  p(-1) = -2+(-3)+(-4)+3 = -6 ✓"},
            ],
            "final_answer": "a = -3, b = -4"
        },
        "marking_scheme": {
            "total_marks": 9,
            "steps": [
                {"step_no": 1, "description": "State and apply Remainder Theorem for both divisors.", "marks": 2},
                {"step_no": 2, "description": "Form equation (1): 2a-b=-2.", "marks": 2},
                {"step_no": 3, "description": "Form equation (2): a+b=-7.", "marks": 2},
                {"step_no": 4, "description": "Solve simultaneous equations: a=-3, b=-4.", "marks": 2},
                {"step_no": 5, "description": "Verify both remainders.", "marks": 1},
            ]
        }
    }


def q17_polynomial_factor_partial():
    return {
        "metadata": {"topic": "polynomials", "solution_type": "partially_correct_factor_theorem", "question_number": 17},
        "reasoning_input": {
            "question_text": (
                "Show that (x-2) is a factor of p(x) = x³ - 6x² + 11x - 6. "
                "Hence factorise p(x) completely."
            ),
            "student_steps": [
                {"step_id": 1,  "content": "p(2) = 8 - 24 + 22 - 6 = 0. ✓ So (x-2) is a factor."},
                {"step_id": 2,  "content": "Divide p(x) by (x-2): x³-6x²+11x-6 ÷ (x-2)"},
                {"step_id": 3,  "content": "x³÷x = x². x²(x-2) = x³-2x². Remainder: -4x²+11x-6"},
                {"step_id": 4,  "content": "-4x²÷x = -4x. -4x(x-2) = -4x²+8x. Remainder: 3x-6"},
                {"step_id": 5,  "content": "3x÷x = 3. 3(x-2) = 3x-6. Remainder: 0."},
                {"step_id": 6,  "content": "Quotient: x²-4x+3"},
                {"step_id": 7,  "content": "Student writes: x²-4x+3 = (x-3)(x-1)  [WRONG: should be (x-3)(x-1) — actually correct!]"},
                {"step_id": 8,  "content": "But student does not state the complete factorisation p(x)=(x-2)(x-3)(x-1)."},
            ],
            "final_answer": "Student got quotient right but did not write complete factorisation."
        },
        "marking_scheme": {
            "total_marks": 8,
            "steps": [
                {"step_no": 1, "description": "Show p(2)=0 to verify factor.", "marks": 1},
                {"step_no": 2, "description": "Polynomial division giving x²-4x+3.", "marks": 3},
                {"step_no": 3, "description": "Factorise quadratic to (x-1)(x-3).", "marks": 2},
                {"step_no": 4, "description": "State complete factorisation (x-2)(x-1)(x-3) — not done. 0 marks.", "marks": 0},
            ]
        }
    }


def q18_polynomial_division_wrong():
    return {
        "metadata": {"topic": "polynomials", "solution_type": "wrong_method", "question_number": 18},
        "reasoning_input": {
            "question_text": (
                "Find the quotient and remainder when 2x⁴ - x³ + 3x - 5 is divided by (x² - 2)."
            ),
            "student_steps": [
                {"step_id": 1,  "content": "Student attempts: 2x⁴/x² = 2x². 2x²(x²-2) = 2x⁴-4x²."},
                {"step_id": 2,  "content": "Remainder: -x³ + 4x² + 3x - 5"},
                {"step_id": 3,  "content": "-x³/x² = -x. -x(x²-2) = -x³+2x."},
                {"step_id": 4,  "content": "Remainder: 4x² + x - 5"},
                {"step_id": 5,  "content": "4x²/x² = 4. 4(x²-2) = 4x²-8."},
                {"step_id": 6,  "content": "Remainder: x + 3"},
                {"step_id": 7,  "content": "Student writes quotient = 2x²-x+4, remainder = x+3. [Correct!]"},
                {"step_id": 8,  "content": "But then writes the answer as (2x²-x+4) + (x+3)/(x²-2) without the original polynomial context. Incomplete."},
            ],
            "final_answer": "Quotient = 2x²-x+4, remainder = x+3 (correct but final presentation incomplete)"
        },
        "marking_scheme": {
            "total_marks": 8,
            "steps": [
                {"step_no": 1, "description": "First division step: 2x⁴÷x²=2x².", "marks": 1},
                {"step_no": 2, "description": "Subtract 2x²(x²-2) and get correct remainder -x³+4x²+3x-5.", "marks": 2},
                {"step_no": 3, "description": "Continue division: -x, subtract, remainder 4x²+x-5.", "marks": 2},
                {"step_no": 4, "description": "Final step: 4, remainder x+3.", "marks": 2},
                {"step_no": 5, "description": "State quotient and remainder clearly.", "marks": 1},
            ]
        }
    }


# ===========================================================================
# 7.  MATRICES  (q19 – q21)
# ===========================================================================

def q19_matrix_inverse_perfect():
    return {
        "metadata": {"topic": "matrices", "solution_type": "perfect_inverse", "question_number": 19},
        "reasoning_input": {
            "question_text": (
                "A = [[3, 1], [5, 2]]. Find A⁻¹ and hence solve the system: 3x+y=7, 5x+2y=11."
            ),
            "student_steps": [
                {"step_id": 1,  "content": "det(A) = 3(2) - 1(5) = 6 - 5 = 1"},
                {"step_id": 2,  "content": "A⁻¹ = (1/1)[[2, -1], [-5, 3]] = [[2, -1], [-5, 3]]"},
                {"step_id": 3,  "content": "System: AX = B where B = [7, 11]ᵀ"},
                {"step_id": 4,  "content": "X = A⁻¹B = [[2,-1],[-5,3]] × [7,11]ᵀ"},
                {"step_id": 5,  "content": "x = 2(7) + (-1)(11) = 14 - 11 = 3"},
                {"step_id": 6,  "content": "y = -5(7) + 3(11) = -35 + 33 = -2"},
                {"step_id": 7,  "content": "Check: 3(3)+(-2)=9-2=7 ✓  5(3)+2(-2)=15-4=11 ✓"},
            ],
            "final_answer": "A⁻¹ = [[2,-1],[-5,3]]; x=3, y=-2"
        },
        "marking_scheme": {
            "total_marks": 9,
            "steps": [
                {"step_no": 1, "description": "Calculate det(A) = 1.", "marks": 1},
                {"step_no": 2, "description": "Form A⁻¹ correctly.", "marks": 2},
                {"step_no": 3, "description": "Set up matrix equation AX=B.", "marks": 1},
                {"step_no": 4, "description": "Multiply A⁻¹B to get x=3.", "marks": 2},
                {"step_no": 5, "description": "Get y=-2.", "marks": 2},
                {"step_no": 6, "description": "Verify both equations.", "marks": 1},
            ]
        }
    }


def q20_matrix_determinant_error():
    return {
        "metadata": {"topic": "matrices", "solution_type": "wrong_determinant", "question_number": 20},
        "reasoning_input": {
            "question_text": (
                "Find the inverse of B = [[4, 3], [2, 1]]."
            ),
            "student_steps": [
                {"step_id": 1,  "content": "det(B) = 4(1) - 3(2) = 4 - 6 = -2"},
                {"step_id": 2,  "content": "Student writes: B⁻¹ = (1/(-2)) × [[1, 3], [-2, 4]]"},
                {"step_id": 3,  "content": "ERROR: Student forgot to swap a and d. Should be [[1,-3],[-2,4]] but wrote [[1,3],[-2,4]]."},
                {"step_id": 4,  "content": "B⁻¹ = [[-1/2, -3/2], [1, -2]]  (incorrect)"},
            ],
            "final_answer": "[[-1/2, -3/2], [1, -2]] (incorrect — sign error in adjugate)"
        },
        "marking_scheme": {
            "total_marks": 6,
            "steps": [
                {"step_no": 1, "description": "Calculate det = -2 correctly.", "marks": 2},
                {"step_no": 2, "description": "Form adjugate: swap a,d and negate b,c — student got sign wrong for b. 0 marks.", "marks": 0},
                {"step_no": 3, "description": "Multiply by 1/det = -1/2 — applied but to wrong matrix.", "marks": 1},
                {"step_no": 4, "description": "Final correct B⁻¹ not reached.", "marks": 0},
            ]
        }
    }


def q21_matrix_system_partial():
    return {
        "metadata": {"topic": "matrices", "solution_type": "partially_correct_system", "question_number": 21},
        "reasoning_input": {
            "question_text": (
                "Using matrices, solve: 2x - y = 3, x + 3y = 10."
            ),
            "student_steps": [
                {"step_id": 1,  "content": "Write in matrix form: [[2,-1],[1,3]] × [x,y]ᵀ = [3,10]ᵀ"},
                {"step_id": 2,  "content": "det = 2(3) - (-1)(1) = 6 + 1 = 7"},
                {"step_id": 3,  "content": "Inverse = (1/7)[[3,1],[-1,2]]"},
                {"step_id": 4,  "content": "x = (1/7)(3×3 + 1×10) = (1/7)(19) = 19/7   ← WRONG: should be 3(3)+1(10)=19, x=19/7"},
                {"step_id": 5,  "content": "y = (1/7)(-1×3 + 2×10) = (1/7)(17) = 17/7"},
                {"step_id": 6,  "content": "Student does not verify. Also does not check if 19/7 and 17/7 satisfy original equations."},
            ],
            "final_answer": "x = 19/7, y = 17/7 (correct values, no verification)"
        },
        "marking_scheme": {
            "total_marks": 8,
            "steps": [
                {"step_no": 1, "description": "Set up matrix equation correctly.", "marks": 1},
                {"step_no": 2, "description": "Calculate det = 7.", "marks": 1},
                {"step_no": 3, "description": "Write inverse correctly.", "marks": 2},
                {"step_no": 4, "description": "Multiply to find x = 19/7.", "marks": 2},
                {"step_no": 5, "description": "Find y = 17/7.", "marks": 1},
                {"step_no": 6, "description": "Verify — not done. 0 marks.", "marks": 0},
            ]
        }
    }


# ===========================================================================
# 8.  SEQUENCES & SERIES  (q22 – q24)
# ===========================================================================

def q22_arithmetic_series_perfect():
    return {
        "metadata": {"topic": "sequences_series", "solution_type": "perfect_arithmetic", "question_number": 22},
        "reasoning_input": {
            "question_text": (
                "The 4th term of an arithmetic progression is 17 and the 10th term is 35. "
                "Find (a) the common difference, (b) the first term, (c) the sum of the first 20 terms."
            ),
            "student_steps": [
                {"step_id": 1,  "content": "aₙ = a + (n-1)d. a₄ = a + 3d = 17  ... (1)"},
                {"step_id": 2,  "content": "a₁₀ = a + 9d = 35   ... (2)"},
                {"step_id": 3,  "content": "(2)-(1): 6d = 18 → d = 3"},
                {"step_id": 4,  "content": "From (1): a + 9 = 17 → a = 8"},
                {"step_id": 5,  "content": "S₂₀ = (20/2)[2(8) + 19(3)] = 10[16 + 57] = 10(73) = 730"},
            ],
            "final_answer": "d = 3, a = 8, S₂₀ = 730"
        },
        "marking_scheme": {
            "total_marks": 8,
            "steps": [
                {"step_no": 1, "description": "Form two equations using AP formula.", "marks": 2},
                {"step_no": 2, "description": "Solve for d = 3.", "marks": 2},
                {"step_no": 3, "description": "Find a = 8.", "marks": 1},
                {"step_no": 4, "description": "Apply sum formula Sₙ = n/2[2a+(n-1)d].", "marks": 2},
                {"step_no": 5, "description": "Compute S₂₀ = 730.", "marks": 1},
            ]
        }
    }


def q23_geometric_series_partial():
    return {
        "metadata": {"topic": "sequences_series", "solution_type": "partially_correct_geometric", "question_number": 23},
        "reasoning_input": {
            "question_text": (
                "A geometric progression has first term 5 and common ratio 3. "
                "Find the least value of n such that the sum of the first n terms exceeds 10000."
            ),
            "student_steps": [
                {"step_id": 1,  "content": "Sₙ = 5(3ⁿ - 1)/(3-1) = 5(3ⁿ-1)/2"},
                {"step_id": 2,  "content": "5(3ⁿ-1)/2 > 10000 → 3ⁿ-1 > 4000 → 3ⁿ > 4001"},
                {"step_id": 3,  "content": "n log 3 > log 4001"},
                {"step_id": 4,  "content": "n > log(4001)/log(3) = 3.6021/0.4771 = 7.55"},
                {"step_id": 5,  "content": "Student writes n > 7.55, so n = 7.   ← WRONG: should be n = 8."},
            ],
            "final_answer": "n = 7 (wrong — correct is n = 8)"
        },
        "marking_scheme": {
            "total_marks": 8,
            "steps": [
                {"step_no": 1, "description": "Apply GP sum formula correctly.", "marks": 2},
                {"step_no": 2, "description": "Form inequality 3ⁿ > 4001.", "marks": 2},
                {"step_no": 3, "description": "Take logarithms and solve n > 7.55.", "marks": 2},
                {"step_no": 4, "description": "Round up to n=8 since n must be integer — student rounded down. 0 marks.", "marks": 0},
            ]
        }
    }


def q24_sum_infinity_wrong_condition():
    return {
        "metadata": {"topic": "sequences_series", "solution_type": "wrong_convergence_condition", "question_number": 24},
        "reasoning_input": {
            "question_text": (
                "A geometric series has first term 12 and common ratio r. "
                "The sum to infinity is 20. Find r. State the condition for the sum to infinity to exist."
            ),
            "student_steps": [
                {"step_id": 1,  "content": "S∞ = a/(1-r). 12/(1-r) = 20"},
                {"step_id": 2,  "content": "12 = 20(1-r) = 20 - 20r → 20r = 8 → r = 0.4"},
                {"step_id": 3,  "content": "Student writes: condition is r > 0.   ← WRONG: should be |r| < 1."},
            ],
            "final_answer": "r = 0.4 (correct), condition r>0 (wrong — should be |r|<1)"
        },
        "marking_scheme": {
            "total_marks": 6,
            "steps": [
                {"step_no": 1, "description": "Use S∞ = a/(1-r) = 20.", "marks": 1},
                {"step_no": 2, "description": "Solve for r = 0.4.", "marks": 3},
                {"step_no": 3, "description": "State correct convergence condition |r| < 1 — student wrote r>0. 0 marks.", "marks": 0},
            ]
        }
    }


# ===========================================================================
# 9.  COMPLEX NUMBERS  (q25 – q27)
# ===========================================================================

def q25_complex_argand_perfect():
    return {
        "metadata": {"topic": "complex_numbers", "solution_type": "perfect_argand", "question_number": 25},
        "reasoning_input": {
            "question_text": (
                "Given z = 3 - 4i, find (a) |z|, (b) arg(z), (c) the complex conjugate z̄, "
                "(d) z × z̄."
            ),
            "student_steps": [
                {"step_id": 1,  "content": "(a) |z| = √(3² + (-4)²) = √(9+16) = √25 = 5"},
                {"step_id": 2,  "content": "(b) arg(z): reference angle = arctan(4/3) ≈ 53.13°"},
                {"step_id": 3,  "content": "z is in 4th quadrant (Re>0, Im<0), so arg(z) = -53.13° ≈ -0.927 rad"},
                {"step_id": 4,  "content": "(c) z̄ = 3 + 4i"},
                {"step_id": 5,  "content": "(d) z × z̄ = (3-4i)(3+4i) = 9 + 12i - 12i - 16i² = 9 + 16 = 25"},
                {"step_id": 6,  "content": "Alternatively, z × z̄ = |z|² = 5² = 25 ✓"},
            ],
            "final_answer": "|z|=5, arg(z)≈-0.927 rad, z̄=3+4i, zz̄=25"
        },
        "marking_scheme": {
            "total_marks": 8,
            "steps": [
                {"step_no": 1, "description": "Find modulus |z| = 5.", "marks": 2},
                {"step_no": 2, "description": "Find arg(z) in correct quadrant.", "marks": 2},
                {"step_no": 3, "description": "State conjugate z̄ = 3+4i.", "marks": 1},
                {"step_no": 4, "description": "Compute z×z̄ = 25.", "marks": 2},
                {"step_no": 5, "description": "Verify using |z|².", "marks": 1},
            ]
        }
    }


def q26_complex_de_moivre_perfect():
    return {
        "metadata": {"topic": "complex_numbers", "solution_type": "perfect_de_moivre", "question_number": 26},
        "reasoning_input": {
            "question_text": (
                "Use De Moivre's theorem to find (1 + i)⁸. "
                "Express your answer in the form a + bi."
            ),
            "student_steps": [
                {"step_id": 1,  "content": "Write 1+i in polar form. |1+i| = √2, arg(1+i) = π/4"},
                {"step_id": 2,  "content": "1+i = √2 (cos(π/4) + i sin(π/4))"},
                {"step_id": 3,  "content": "(1+i)⁸ = (√2)⁸ (cos(8π/4) + i sin(8π/4))"},
                {"step_id": 4,  "content": "= 2⁴ (cos(2π) + i sin(2π))"},
                {"step_id": 5,  "content": "= 16 (1 + 0i)"},
                {"step_id": 6,  "content": "= 16"},
                {"step_id": 7,  "content": "Verification: (1+i)² = 2i. (2i)² = -4. (-4)² = 16 ✓"},
            ],
            "final_answer": "16"
        },
        "marking_scheme": {
            "total_marks": 10,
            "steps": [
                {"step_no": 1, "description": "Find modulus √2 and argument π/4.", "marks": 2},
                {"step_no": 2, "description": "Write in polar form correctly.", "marks": 1},
                {"step_no": 3, "description": "Apply De Moivre's theorem.", "marks": 2},
                {"step_no": 4, "description": "Compute (√2)⁸ = 16.", "marks": 2},
                {"step_no": 5, "description": "Evaluate cos(2π)+i sin(2π) = 1.", "marks": 2},
                {"step_no": 6, "description": "State answer 16 with verification.", "marks": 1},
            ]
        }
    }


def q27_complex_modulus_argument_partial():
    return {
        "metadata": {"topic": "complex_numbers", "solution_type": "partially_correct_argument", "question_number": 27},
        "reasoning_input": {
            "question_text": (
                "Find the modulus and argument of z = (2 + 2i) / (1 - i)."
            ),
            "student_steps": [
                {"step_id": 1,  "content": "Multiply numerator and denominator by conjugate of denominator: (1+i)"},
                {"step_id": 2,  "content": "z = (2+2i)(1+i) / (1-i)(1+i) = (2+2i+2i+2i²) / (1+1)"},
                {"step_id": 3,  "content": "= (2 + 4i - 2) / 2 = 4i/2 = 2i"},
                {"step_id": 4,  "content": "|z| = |2i| = 2 ✓"},
                {"step_id": 5,  "content": "arg(z): student writes arg(2i) = 0  ← WRONG. Should be π/2."},
            ],
            "final_answer": "|z|=2 (correct), arg(z)=0 (wrong — should be π/2)"
        },
        "marking_scheme": {
            "total_marks": 8,
            "steps": [
                {"step_no": 1, "description": "Multiply by conjugate (1+i)/(1+i).", "marks": 1},
                {"step_no": 2, "description": "Expand numerator correctly.", "marks": 2},
                {"step_no": 3, "description": "Simplify to z = 2i.", "marks": 2},
                {"step_no": 4, "description": "Find |z| = 2.", "marks": 2},
                {"step_no": 5, "description": "Find arg(z) = π/2 — student wrote 0. 0 marks.", "marks": 0},
            ]
        }
    }


# ===========================================================================
# 10.  MISC ALGEBRA  (q28 – q30)
# ===========================================================================

def q28_algebraic_fractions_perfect():
    return {
        "metadata": {"topic": "misc_algebra", "solution_type": "perfect_algebraic_fractions", "question_number": 28},
        "reasoning_input": {
            "question_text": (
                "Simplify: [x² - 4] / [x² - x - 6] ÷ [x + 2] / [x² - 9]."
            ),
            "student_steps": [
                {"step_id": 1,  "content": "Factorise all expressions:"},
                {"step_id": 2,  "content": "x²-4 = (x-2)(x+2)"},
                {"step_id": 3,  "content": "x²-x-6 = (x-3)(x+2)"},
                {"step_id": 4,  "content": "x²-9 = (x-3)(x+3)"},
                {"step_id": 5,  "content": "Division becomes: [(x-2)(x+2)] / [(x-3)(x+2)] × [(x-3)(x+3)] / [(x+2)]"},
                {"step_id": 6,  "content": "Cancel (x+2) from numerator and denominator:"},
                {"step_id": 7,  "content": "= [(x-2)(x-3)(x+3)] / [(x-3)(x+2)]"},
                {"step_id": 8,  "content": "Cancel (x-3): = (x-2)(x+3) / (x+2)"},
                {"step_id": 9,  "content": "= (x²+x-6) / (x+2)"},
            ],
            "final_answer": "(x-2)(x+3)/(x+2) or (x²+x-6)/(x+2)"
        },
        "marking_scheme": {
            "total_marks": 8,
            "steps": [
                {"step_no": 1, "description": "Factorise all four expressions correctly.", "marks": 3},
                {"step_no": 2, "description": "Invert and multiply (flip second fraction for division).", "marks": 1},
                {"step_no": 3, "description": "Cancel common factors systematically.", "marks": 3},
                {"step_no": 4, "description": "State final simplified form.", "marks": 1},
            ]
        }
    }


def q29_simultaneous_nonlinear_partial():
    return {
        "metadata": {"topic": "misc_algebra", "solution_type": "partially_correct_simultaneous", "question_number": 29},
        "reasoning_input": {
            "question_text": (
                "Solve the simultaneous equations: y = x² - 3x + 2 and y = 2x - 4."
            ),
            "student_steps": [
                {"step_id": 1,  "content": "x² - 3x + 2 = 2x - 4"},
                {"step_id": 2,  "content": "x² - 5x + 6 = 0"},
                {"step_id": 3,  "content": "(x-2)(x-3) = 0"},
                {"step_id": 4,  "content": "x = 2 or x = 3"},
                {"step_id": 5,  "content": "Student only finds x values and writes x=2, x=3 without finding y values."},
            ],
            "final_answer": "x=2, x=3 (y values not found)"
        },
        "marking_scheme": {
            "total_marks": 8,
            "steps": [
                {"step_no": 1, "description": "Equate y expressions and rearrange.", "marks": 1},
                {"step_no": 2, "description": "Form x²-5x+6=0.", "marks": 1},
                {"step_no": 3, "description": "Factorise and solve x=2, x=3.", "marks": 3},
                {"step_no": 4, "description": "Find y: when x=2, y=0; when x=3, y=2 — not done. 0 marks.", "marks": 0},
                {"step_no": 5, "description": "State both solution pairs (2,0) and (3,2) — not done. 0 marks.", "marks": 0},
            ]
        }
    }


def q30_exponential_equations_wrong():
    return {
        "metadata": {"topic": "misc_algebra", "solution_type": "wrong_method", "question_number": 30},
        "reasoning_input": {
            "question_text": (
                "Solve: 4^x - 6(2^x) + 8 = 0."
            ),
            "student_steps": [
                {"step_id": 1,  "content": "Student does NOT substitute u = 2^x."},
                {"step_id": 2,  "content": "Student tries: 4^x - 6(2^x) + 8 = 0 → 2^(2x) = 6(2^x) - 8"},
                {"step_id": 3,  "content": "Takes log: 2x log 2 = log(6·2^x - 8)  ← Cannot simplify RHS."},
                {"step_id": 4,  "content": "Student is stuck and cannot proceed."},
                {"step_id": 5,  "content": "Correct approach: let u = 2^x → u² - 6u + 8 = 0 → (u-2)(u-4)=0"},
                {"step_id": 6,  "content": "u=2 → 2^x=2 → x=1. u=4 → 2^x=4 → x=2."},
            ],
            "final_answer": "Student could not solve — wrong approach taken. Correct: x=1 or x=2."
        },
        "marking_scheme": {
            "total_marks": 8,
            "steps": [
                {"step_no": 1, "description": "Recognise substitution u=2^x to convert to quadratic — not done. 0 marks.", "marks": 0},
                {"step_no": 2, "description": "Form u²-6u+8=0 — not reached.", "marks": 0},
                {"step_no": 3, "description": "Factorise (u-2)(u-4)=0 — not reached.", "marks": 0},
                {"step_no": 4, "description": "Find u=2 and u=4 — not reached.", "marks": 0},
                {"step_no": 5, "description": "Convert back: x=1 and x=2 — not reached.", "marks": 0},
            ]
        }
    }


# ===========================================================================
# REGISTRY
# ===========================================================================

CASES = [
    # (subfolder, filename, generator_function)
    ("induction",         "q01_induction_perfect.json",                       q01_induction_perfect),
    ("induction",         "q02_induction_missing_hypothesis.json",             q02_induction_missing_hypothesis),
    ("induction",         "q03_induction_wrong_base_case.json",                q03_induction_wrong_base_case),
    ("partial_fractions", "q04_partial_fractions_perfect.json",                q04_partial_fractions_perfect),
    ("partial_fractions", "q05_partial_fractions_partially_correct.json",      q05_partial_fractions_partially_correct),
    ("partial_fractions", "q06_partial_fractions_repeated_factor_perfect.json",q06_partial_fractions_repeated_factor_perfect),
    ("binomial",          "q07_binomial_perfect.json",                         q07_binomial_perfect),
    ("binomial",          "q08_binomial_missing_validity.json",                q08_binomial_missing_validity),
    ("binomial",          "q09_binomial_wrong_approximation.json",             q09_binomial_approximation_wrong),
    ("logarithms",        "q10_log_equations_perfect.json",                    q10_log_perfect),
    ("logarithms",        "q11_log_equations_domain_error.json",               q11_log_domain_error),
    ("logarithms",        "q12_log_change_base_partial.json",                  q12_log_change_base_partial),
    ("quadratic",         "q13_quadratic_completing_square_perfect.json",      q13_quadratic_completing_square_perfect),
    ("quadratic",         "q14_quadratic_discriminant_sign_error.json",        q14_quadratic_discriminant_sign_error),
    ("quadratic",         "q15_quadratic_formula_partial.json",                q15_quadratic_formula_partial),
    ("polynomials",       "q16_polynomial_remainder_theorem_perfect.json",     q16_polynomial_remainder_perfect),
    ("polynomials",       "q17_polynomial_factor_theorem_partial.json",        q17_polynomial_factor_partial),
    ("polynomials",       "q18_polynomial_division_wrong.json",                q18_polynomial_division_wrong),
    ("matrices",          "q19_matrix_inverse_perfect.json",                   q19_matrix_inverse_perfect),
    ("matrices",          "q20_matrix_determinant_error.json",                 q20_matrix_determinant_error),
    ("matrices",          "q21_matrix_system_partial.json",                    q21_matrix_system_partial),
    ("sequences_series",  "q22_arithmetic_series_perfect.json",                q22_arithmetic_series_perfect),
    ("sequences_series",  "q23_geometric_series_partial.json",                 q23_geometric_series_partial),
    ("sequences_series",  "q24_sum_to_infinity_wrong_condition.json",          q24_sum_infinity_wrong_condition),
    ("complex_numbers",   "q25_complex_argand_perfect.json",                   q25_complex_argand_perfect),
    ("complex_numbers",   "q26_complex_de_moivre_perfect.json",               q26_complex_de_moivre_perfect),
    ("complex_numbers",   "q27_complex_modulus_argument_partial.json",         q27_complex_modulus_argument_partial),
    ("misc_algebra",      "q28_algebraic_fractions_perfect.json",              q28_algebraic_fractions_perfect),
    ("misc_algebra",      "q29_simultaneous_nonlinear_partial.json",           q29_simultaneous_nonlinear_partial),
    ("misc_algebra",      "q30_exponential_equations_wrong.json",              q30_exponential_equations_wrong),
]


if __name__ == "__main__":
    print(f"\n📁  Writing to: {BASE_DIR.resolve()}\n")
    for subfolder, filename, fn in CASES:
        write(subfolder, filename, fn())
    total = len(CASES)
    print(f"\n✅  Done! {total} test cases generated in al_algebra_cases/")
