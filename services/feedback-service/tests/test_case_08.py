"""
Sample case 08 — matrix addition.

Question : Find A + B, where A = [[1, 2], [3, 4]] and B = [[5, 6], [7, 8]]
Steps    : 3 steps, mixed (1.5 / 2.0)
Exercises: matrix-valued expressions — the method is right all the way
           through, only the final entry 4 + 8 is written as 13 instead of 12,
           so the feedback has to name that one element rather than the method

Run from the feedback-service directory:
    python tests/test_case_08.py                  # ZeroGPU Space (default)
    python tests/test_case_08.py --backend local  # Colab T4 + LoRA adapter

See tests/case_runner.py for the backend options and the assertions.
"""
from case_runner import main

if __name__ == "__main__":
    main(8)
