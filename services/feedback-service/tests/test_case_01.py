"""
Sample case 01 — linear equation with brackets.

Question : Solve 2(x + 3) = 14
Steps    : 3 steps, all correct (3.0 / 3.0)
Exercises: the full-marks path — the model must not invent MISSING/DEDUCTION text

Run from the feedback-service directory:
    python tests/test_case_01.py                  # ZeroGPU Space (default)
    python tests/test_case_01.py --backend local  # Colab T4 + LoRA adapter

See tests/case_runner.py for the backend options and the assertions.
"""
from case_runner import main

if __name__ == "__main__":
    main(1)
