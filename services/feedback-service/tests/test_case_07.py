"""
Sample case 07 — logarithmic equation (Module 02 output).

Question : Solve log_2(x - 1) + log_2(x - 3) = 3 where x > 3
Steps    : 8 steps — 6 correct, 2 partial (6.0 / 6.0)
Exercises: the longest answer shape the service has to handle, plus a
           domain-restriction step at the end

Run from the feedback-service directory:
    python tests/test_case_07.py                  # ZeroGPU Space (default)
    python tests/test_case_07.py --backend local  # Colab T4 + LoRA adapter

See tests/case_runner.py for the backend options and the assertions.
"""
from case_runner import main

if __name__ == "__main__":
    main(7)
