"""
Sample case 07 — difference of squares.

Question : Factorise 25x^2 - 49
Steps    : 2 steps — correct, partial (1.5 / 2.0)
Exercises: the shortest answer shape the service has to handle

Run from the feedback-service directory:
    python tests/test_case_07.py                  # ZeroGPU Space (default)
    python tests/test_case_07.py --backend local  # Colab T4 + LoRA adapter

See tests/case_runner.py for the backend options and the assertions.
"""
from case_runner import main

if __name__ == "__main__":
    main(7)
