"""
Sample case 10 — remainder theorem.

Question : Find the remainder when x^3 - 4x^2 + 5x - 2 is divided by (x - 3)
Steps    : 4 steps, mixed (3.0 / 4.0)
Exercises: Module 02's legacy is_correct bool instead of validity — the payload
           has no "validity" key at all, so StepResult.derive_validity in
           app/models/schemas.py has to fill it in

Run from the feedback-service directory:
    python tests/test_case_10.py                  # ZeroGPU Space (default)
    python tests/test_case_10.py --backend local  # Colab T4 + LoRA adapter

See tests/case_runner.py for the backend options and the assertions.
"""
from case_runner import main

if __name__ == "__main__":
    main(10)
