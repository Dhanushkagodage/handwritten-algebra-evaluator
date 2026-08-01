"""
Sample case 09 — simultaneous equations by substitution.

Question : Solve y = x + 1 and 2x + 3y = 13
Steps    : 4 steps, mixed (2.5 / 4.0)
Exercises: the optional error_description field on student steps, which Module 02
           may supply and which feeds into the prompt

Run from the feedback-service directory:
    python tests/test_case_09.py                  # ZeroGPU Space (default)
    python tests/test_case_09.py --backend local  # Colab T4 + LoRA adapter

See tests/case_runner.py for the backend options and the assertions.
"""
from case_runner import main

if __name__ == "__main__":
    main(9)
