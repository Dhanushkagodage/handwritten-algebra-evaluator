"""
Sample case 05 — linear inequality.

Question : Solve 4 - 3x < 13
Steps    : 3 steps, incorrect at the flip (2.0 / 3.0)
Exercises: the classic misconception — not reversing < when dividing by a negative

Run from the feedback-service directory:
    python tests/test_case_05.py                  # ZeroGPU Space (default)
    python tests/test_case_05.py --backend local  # Colab T4 + LoRA adapter

See tests/case_runner.py for the backend options and the assertions.
"""
from case_runner import main

if __name__ == "__main__":
    main(5)
