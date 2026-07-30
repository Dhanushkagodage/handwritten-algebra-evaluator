from pathlib import Path
import base64
import json
import os
import re

from openai import OpenAI
from dotenv import load_dotenv


DEFAULT_MODEL = "gpt-5.6"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")


def encode_image_as_data_url(image_path: str) -> str:
    """Read an image file and return a base64 data URL."""
    image_file = Path(image_path)
    suffix = image_file.suffix.lower()
    media_type = "image/png" if suffix == ".png" else "image/jpeg"

    with image_file.open("rb") as file:
        encoded = base64.b64encode(file.read()).decode("utf-8")

    return f"data:{media_type};base64,{encoded}"


def extract_reasoning_input_with_openai(image_path: str, question_text: str = "") -> dict:
    """
    Use an OpenAI vision-capable model to extract Module 01 output directly.

    This is closer to the behavior users see in ChatGPT with image uploads.
    """
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set. Add it to your environment before using OpenAI vision OCR.")

    model = os.getenv("OPENAI_VISION_MODEL", DEFAULT_MODEL)
    client = OpenAI(api_key=api_key)
    image_data_url = encode_image_as_data_url(image_path)
    prompt = build_extraction_prompt(question_text)

    response = client.responses.create(
        model=model,
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt},
                    {"type": "input_image", "image_url": image_data_url, "detail": "original"},
                ],
            }
        ],
    )

    return parse_reasoning_json(response.output_text)


def extract_reasoning_input_from_images_with_openai(image_paths: list[str], question_text: str = "") -> dict:
    """
    Use a vision-capable model to extract one answer attempt from ordered page images.

    This is used when the same student's answer spans more than one photo.
    """
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set. Add it to your environment before using OpenAI vision OCR.")

    if not image_paths:
        raise ValueError("At least one image path is required.")

    model = os.getenv("OPENAI_VISION_MODEL", DEFAULT_MODEL)
    client = OpenAI(api_key=api_key)
    prompt = build_extraction_prompt(question_text, multiple_ordered_images=True)
    content = [{"type": "input_text", "text": prompt}]

    for index, image_path in enumerate(image_paths, start=1):
        content.append({"type": "input_text", "text": f"Page image {index} of {len(image_paths)}."})
        content.append(
            {
                "type": "input_image",
                "image_url": encode_image_as_data_url(image_path),
                "detail": "original",
            }
        )

    response = client.responses.create(
        model=model,
        input=[
            {
                "role": "user",
                "content": content,
            }
        ],
    )

    return parse_reasoning_json(response.output_text)


def build_extraction_prompt(question_text: str = "", multiple_ordered_images: bool = False) -> str:
    """Prompt the vision model to produce only the JSON needed by Module 02."""
    question_instruction = (
        f'Use this question_text exactly if it is relevant: "{question_text.strip()}".'
        if question_text.strip()
        else "If questions are visible in the image, transcribe each question into its own question_text. Otherwise use an empty string."
    )

    page_instruction = (
        "The uploaded images are ordered pages/photos of the same student's answer. "
        "Read them in page order and combine continuing work for the same question into one object."
        if multiple_ordered_images
        else ""
    )

    return f"""
You are Module 01: OCR + Input Understanding for handwritten A/L algebra answers.

Extract what the student wrote from the image. Do not mark the answer. Do not correct mathematical mistakes.
Preserve the student's working steps in order.
If the image contains more than one independent question or answer attempt, split them into separate objects.
{page_instruction}

{question_instruction}

Return only valid JSON in this exact shape:
{{
  "reasoning_inputs": [
    {{
      "question_id": "Q1",
      "question_text": "",
      "student_steps": [
        {{"step_id": 1, "content": ""}}
      ],
      "final_answer": ""
    }}
  ]
}}

Rules:
- Use plain text math such as x^2, (x - 1)^2, A/(x - 1), sqrt(31).
- Treat a new visible "Solve", "Find", question number, or clearly separate answer block as a new question.
- Do not merge two different questions into one question_text.
- Do not move steps from one question into another question.
- Keep crossed-out work only if it is part of the student's visible reasoning; otherwise ignore it.
- Put the last clear result for each question as that question's final_answer.
- Do not include bounding boxes, OCR confidence, backend names, explanations, or markdown.
""".strip()


def parse_reasoning_json(text: str) -> dict:
    """Parse the model response and validate the Module 02 input shape."""
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?", "", cleaned).strip()
    cleaned = re.sub(r"```$", "", cleaned).strip()

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)

        if not match:
            raise

        data = json.loads(match.group(0))

    if "reasoning_inputs" in data:
        raw_inputs = data["reasoning_inputs"]
    elif "reasoning_input" in data:
        raw_inputs = [data["reasoning_input"]]
    else:
        raise ValueError("OpenAI vision OCR response did not include reasoning_inputs.")

    if not isinstance(raw_inputs, list):
        raw_inputs = [raw_inputs]

    cleaned_inputs = []

    for question_index, reasoning_input in enumerate(raw_inputs, start=1):
        if not isinstance(reasoning_input, dict):
            continue

        cleaned_steps = []

        for step_index, step in enumerate(reasoning_input.get("student_steps", []), start=1):
            if isinstance(step, dict):
                content = str(step.get("content", "")).strip()
            else:
                content = str(step).strip()

            if content:
                cleaned_steps.append({"step_id": len(cleaned_steps) + 1, "content": content})

        cleaned_inputs.append(
            {
                "question_id": str(reasoning_input.get("question_id", f"Q{question_index}")).strip()
                or f"Q{question_index}",
                "question_text": str(reasoning_input.get("question_text", "")).strip(),
                "student_steps": cleaned_steps,
                "final_answer": str(reasoning_input.get("final_answer", "")).strip(),
            }
        )

    if not cleaned_inputs:
        raise ValueError("OpenAI vision OCR response did not include any valid question attempts.")

    if len(cleaned_inputs) == 1:
        single_input = dict(cleaned_inputs[0])
        single_input.pop("question_id", None)
        return {"reasoning_input": single_input}

    return {"reasoning_inputs": cleaned_inputs}
