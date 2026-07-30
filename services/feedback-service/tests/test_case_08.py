"""
Sample case 08 — index laws.

Question : Simplify (2x^3 * y^2)^2 / (4x^4 * y)
Steps    : 5 steps, mixed (3.0 / 5.0)
Exercises: the longest chain — pressure-tests the 300 max_new_tokens budget

Run from the feedback-service directory:
    python tests/test_case_08.py                  # ZeroGPU Space (default)
    python tests/test_case_08.py --backend local  # Colab T4 + LoRA adapter

See tests/case_runner.py for the backend options and the assertions.
"""
from case_runner import main

if __name__ == "__main__":
    main(8)
