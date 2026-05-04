"""
Sample test inputs and expected outputs for the /evaluate endpoint.

Run manually with:
    python tests/test_evaluate.py

Or with curl:
    curl -X POST http://localhost:8002/api/v1/evaluate \
         -H "Content-Type: application/json" \
         -d @tests/sample_request.json
"""
import json
import sys

# ── Sample Request ─────────────────────────────────────────────────────────────
SAMPLE_REQUEST = {
    "reasoning_input": {
        "question_text": "Solve the quadratic equation: x² - 5x + 6 = 0",
        "student_steps": [
            {"step_id": 1, "content": "x² - 5x + 6 = 0"},
            {"step_id": 2, "content": "(x - 2)(x - 3) = 0"},
            {"step_id": 3, "content": "x = 2 or x = 3"}
        ],
        "final_answer": "x = 2 or x = 3"
    },
    "marking_scheme": {
        "total_marks": 5,
        "steps": [
            {"step_no": 1, "description": "Correctly write the standard form of the equation", "marks": 1},
            {"step_no": 2, "description": "Correctly factorise the quadratic into two linear factors", "marks": 2},
            {"step_no": 3, "description": "State both correct solutions", "marks": 2}
        ]
    }
}

# ── Sample Expected Output (reference) ────────────────────────────────────────
EXPECTED_OUTPUT_SHAPE = {
    "steps_analysis": [
        {
            "step_id": 1,
            "validity": True,
            "status": "correct",
            "method": "factorization",
            "matched_scheme_step": 1,
            "match_score": 1.0,
            "marks_awarded": 1.0,
            "max_marks": 1.0,
            "reason": "Standard form correctly stated",
            "confidence": 0.95
        },
        {
            "step_id": 2,
            "validity": True,
            "status": "correct",
            "method": "factorization",
            "matched_scheme_step": 2,
            "match_score": 1.0,
            "marks_awarded": 2.0,
            "max_marks": 2.0,
            "reason": "Correct factorisation: (x-2)(x-3) = 0",
            "confidence": 0.98
        },
        {
            "step_id": 3,
            "validity": True,
            "status": "correct",
            "method": "factorization",
            "matched_scheme_step": 3,
            "match_score": 1.0,
            "marks_awarded": 2.0,
            "max_marks": 2.0,
            "reason": "Both roots x=2 and x=3 correctly stated",
            "confidence": 0.99
        }
    ],
    "total_marks": 5.0,
    "max_marks": 5.0,
    "percentage": 100.0,
    "summary": "Excellent work. The student correctly factorised the quadratic and identified both roots accurately.",
    "method_feedback": "Factorization is an efficient and appropriate method for this problem.",
    "missing_steps_feedback": None
}


if __name__ == "__main__":
    try:
        import httpx
    except ImportError:
        print("Install httpx: pip install httpx")
        sys.exit(1)

    url = "http://localhost:8002/api/v1/evaluate"
    print(f"\n{'='*60}")
    print(f"Sending POST {url}")
    print(f"{'='*60}")
    print("\nRequest body:")
    print(json.dumps(SAMPLE_REQUEST, indent=2))

    response = httpx.post(url, json=SAMPLE_REQUEST, timeout=60.0)
    print(f"\n{'='*60}")
    print(f"Response [{response.status_code}]:")
    print(f"{'='*60}")
    if response.is_success:
        print(json.dumps(response.json(), indent=2))
    else:
        print(f"ERROR: {response.text}")
        sys.exit(1)
