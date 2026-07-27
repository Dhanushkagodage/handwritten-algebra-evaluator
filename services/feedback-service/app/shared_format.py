"""Single source of truth for the step-feedback output format, shared by
training (app/training/dataset.py) and inference (app/agents/feedback_generator.py)
so the two can never drift out of sync.
"""

FORMAT_INSTRUCTION = (
    "\nFor each step, respond in this exact format:\n"
    "=== STEP N [CORRECT/PARTIAL/INCORRECT] ===\n"
    "CORRECT: <what the student did correctly, or method acknowledgement>\n"
    "MISSING: <what was wrong or missing — only for INCORRECT or PARTIAL>\n"
    "DEDUCTION: <why marks were reduced — only for INCORRECT or PARTIAL>\n"
    "IMPROVE: <specific actionable tip for the student>\n"
)
