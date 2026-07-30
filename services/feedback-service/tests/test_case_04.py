"""
Sample case 04 — completing the square.

Question : Solve x^2 + 6x + 2 = 0 by completing the square
Steps    : 4 steps, partial-heavy (2.0 / 4.0)
Exercises: 'partial' validity handling — the student dropped the +- on the root

Run from the feedback-service directory:
    python tests/test_case_04.py                  # ZeroGPU Space (default)
    python tests/test_case_04.py --backend local  # Colab T4 + LoRA adapter

See tests/case_runner.py for the backend options and the assertions.
"""
from case_runner import main

if __name__ == "__main__":
    main(4)
