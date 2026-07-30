"""
Sample case 03 — simultaneous equations by elimination.

Question : Solve 3x + 2y = 16 and x - 2y = 0
Steps    : 4 steps — 3 correct, then 1 incorrect (3.0 / 4.0)
Exercises: a sign error that appears only at the very last step

Run from the feedback-service directory:
    python tests/test_case_03.py                  # ZeroGPU Space (default)
    python tests/test_case_03.py --backend local  # Colab T4 + LoRA adapter

See tests/case_runner.py for the backend options and the assertions.
"""
from case_runner import main

if __name__ == "__main__":
    main(3)
