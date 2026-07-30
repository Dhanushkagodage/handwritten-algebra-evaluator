from correction import detect_step_type


QUESTION_KEYWORDS = (
    "solve",
    "resolve",
    "find",
    "evaluate",
    "simplify",
    "factor",
    "factorise",
    "differentiate",
    "integrate",
    "prove",
    "show",
)


def _recognized_text(value) -> str:
    """Extract plain text from a recognized OCR value."""
    if isinstance(value, dict):
        return str(value.get("text", "")).strip()

    return str(value).strip()


def _looks_like_question(text: str) -> bool:
    """Return True when a line looks like the written question/prompt."""
    lower_text = text.lower()
    return any(keyword in lower_text for keyword in QUESTION_KEYWORDS)


def build_reasoning_input(recognized_steps: list, question_text: str = "") -> dict:
    """
    Build the clean Module 02 input format.

    This output intentionally excludes marking fields, OCR backend metadata,
    bounding boxes, and debug image paths.
    """
    lines = [_recognized_text(step) for step in recognized_steps]
    lines = [line for line in lines if line]

    question_text = question_text.strip()
    working_lines = lines

    if not question_text and lines and _looks_like_question(lines[0]):
        question_text = lines[0]
        working_lines = lines[1:]

    final_answer = working_lines[-1] if working_lines else ""
    step_lines = working_lines[:-1] if final_answer else working_lines

    return {
        "reasoning_input": {
            "question_text": question_text,
            "student_steps": [
                {"step_id": index, "content": content}
                for index, content in enumerate(step_lines, start=1)
            ],
            "final_answer": final_answer,
        }
    }


def build_structured_output(annotation: dict, regions: list, recognized_steps: list) -> dict:
    """
    Build the final JSON structure for one handwritten answer image.

    This function does not mark the answer. It only combines the question
    metadata, cropped step regions, and recognized text into one clean object.
    """
    student_steps = []

    for index, region in enumerate(regions):
        recognized_step = recognized_steps[index] if index < len(recognized_steps) else ""

        if isinstance(recognized_step, dict):
            text = recognized_step.get("text", "")
            raw_latex = recognized_step.get("raw_latex", "")
            ocr_backend = recognized_step.get("backend", "")
            easyocr_text = recognized_step.get("easyocr_text", "")
            ocr_warning = recognized_step.get("ocr_warning", "")
            ocr_error = recognized_step.get("ocr_error", "")
        else:
            text = recognized_step
            raw_latex = ""
            ocr_backend = ""
            easyocr_text = ""
            ocr_warning = ""
            ocr_error = ""

        step = {
            "step_id": region.get("step_id", index + 1),
            "text": text,
            "expression": text,
            "type": detect_step_type(text),
            "bbox": region.get("bbox", []),
            "region_image": region.get("image_path", ""),
            "marks_awarded": 0.0,
        }

        if raw_latex:
            step["raw_latex"] = raw_latex

        if ocr_backend:
            step["ocr_backend"] = ocr_backend

        if easyocr_text and easyocr_text != text:
            step["easyocr_text"] = easyocr_text

        if ocr_warning:
            step["ocr_warning"] = ocr_warning

        if ocr_error:
            step["ocr_error"] = ocr_error

        student_steps.append(step)

    answer_text = "\n".join(step["text"] for step in student_steps if step["text"])

    return {
        "image_id": annotation.get("image_id", ""),
        "question_id": annotation.get("question_id", ""),
        "answer_type": annotation.get("answer_type", ""),
        "question_text": annotation.get("question_text", ""),
        "answer_text": answer_text,
        "student_steps": student_steps,
        "linked_question": annotation.get("question_id", ""),
    }
