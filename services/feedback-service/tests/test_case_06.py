"""
Sample case 06 — simplifying algebraic fractions.

Question : Simplify (x^2 - 9) / (x^2 + 5x + 6)
Steps    : 3 steps, all incorrect (0.0 / 3.0)
Exercises: the zero-marks edge case — every step must carry a deduction reason

Run from the feedback-service directory:
    python tests/test_case_06.py                  # ZeroGPU Space (default)
    python tests/test_case_06.py --backend local  # Colab T4 + LoRA adapter

See tests/case_runner.py for the backend options and the assertions.
"""
from case_runner import main

if __name__ == "__main__":
    main(6)
