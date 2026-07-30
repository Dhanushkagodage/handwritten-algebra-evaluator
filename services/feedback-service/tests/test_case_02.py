"""
Sample case 02 — quadratic formula.

Question : Solve 2x^2 - 5x - 3 = 0 using the quadratic formula
Steps    : 4 steps — correct, partial, correct, incorrect (2.5 / 4.0)
Exercises: the standard mixed case; a sign slip inside the discriminant

Run from the feedback-service directory:
    python tests/test_case_02.py                  # ZeroGPU Space (default)
    python tests/test_case_02.py --backend local  # Colab T4 + LoRA adapter

See tests/case_runner.py for the backend options and the assertions.
"""
from case_runner import main

if __name__ == "__main__":
    main(2)
