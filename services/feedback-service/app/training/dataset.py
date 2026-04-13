"""
Dataset preparation for LoRA fine-tuning.
Converts raw teacher annotations into training format.

Raw format (raw_annotations.json):
[
  {
    "question": "Solve x^2 - 5x + 6 = 0",
    "method": "factorization",
    "marks": 4.5,
    "total_marks": 5,
    "steps": [
      {"step_number": 1, "expression": "x^2 - 5x + 6 = 0", "is_correct": true},
      {"step_number": 2, "expression": "(x-2)(x-3) = 0", "is_correct": true},
      {"step_number": 3, "expression": "x = 2 or x = 4", "is_correct": false}
    ],
    "teacher_feedback": "Step 1 and 2 are correct. In Step 3, x=4 is wrong — it should be x=3."
  }
]
"""

import json
import os
from typing import Dict, List


def format_example(item: Dict) -> Dict:
    """Format a single annotation into prompt-completion format for SFT."""
    steps_text = "\n".join(
        [
            f"Step {s['step_number']}: {s['expression']} | "
            f"{'✓ Correct' if s['is_correct'] else '✗ Incorrect'}"
            for s in item["steps"]
        ]
    )

    prompt = (
        f"<start_of_turn>user\n"
        f"You are an algebra teacher giving feedback on a student's exam answer.\n\n"
        f"Question: {item['question']}\n"
        f"Solution Method: {item['method']}\n"
        f"Marks Awarded: {item['marks']} / {item['total_marks']}\n\n"
        f"Student's Steps:\n{steps_text}\n\n"
        f"Give clear, step-by-step feedback that confirms correct steps, "
        f"explains mistakes simply, and suggests improvements.\n"
        f"<end_of_turn>\n"
        f"<start_of_turn>model\n"
        f"{item['teacher_feedback']}<end_of_turn>"
    )

    return {"text": prompt}


def prepare_dataset(input_file: str, output_file: str) -> List[Dict]:
    """Convert raw annotations to training-ready JSON format."""
    with open(input_file, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    formatted = [format_example(item) for item in raw_data]

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(formatted, f, indent=2, ensure_ascii=False)

    print(f"Prepared {len(formatted)} examples → {output_file}")
    return formatted


def create_sample_annotations(output_file: str):
    """Create sample raw annotations for testing the pipeline."""
    samples = [
        {
            "question": "Solve x^2 - 5x + 6 = 0",
            "method": "factorization",
            "marks": 4.5,
            "total_marks": 5,
            "steps": [
                {"step_number": 1, "expression": "x^2 - 5x + 6 = 0", "is_correct": True},
                {"step_number": 2, "expression": "(x-2)(x-3) = 0", "is_correct": True},
                {"step_number": 3, "expression": "x = 2 or x = 4", "is_correct": False},
            ],
            "teacher_feedback": (
                "Step 1: Good, you correctly identified the equation. "
                "Step 2: Excellent factorization! "
                "Step 3: Almost there — x = 4 is incorrect. "
                "From (x-3) = 0, you get x = 3, not x = 4. "
                "Final answers should be x = 2 or x = 3. "
                "Remember to check by substituting back into the original equation."
            ),
        },
        {
            "question": "Solve 2x + 3 = 7",
            "method": "linear equation",
            "marks": 3,
            "total_marks": 3,
            "steps": [
                {"step_number": 1, "expression": "2x + 3 = 7", "is_correct": True},
                {"step_number": 2, "expression": "2x = 4", "is_correct": True},
                {"step_number": 3, "expression": "x = 2", "is_correct": True},
            ],
            "teacher_feedback": (
                "Step 1: Correct starting point. "
                "Step 2: Good — you correctly subtracted 3 from both sides. "
                "Step 3: Perfect — x = 2 is the correct answer. Well done!"
            ),
        },
    ]

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(samples, f, indent=2, ensure_ascii=False)

    print(f"Sample annotations created → {output_file}")


if __name__ == "__main__":
    create_sample_annotations("app/training/data/raw_annotations.json")
    prepare_dataset(
        input_file="app/training/data/raw_annotations.json",
        output_file="app/training/data/feedback_dataset.json",
    )
