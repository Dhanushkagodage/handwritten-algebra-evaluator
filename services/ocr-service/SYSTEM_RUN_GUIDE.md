# Fully Working Module 01 System

This service accepts a handwritten A/L algebra answer image and returns the
structured JSON required by the next module.

## Main Endpoint

```text
POST /extract
```

Use this when the answer is in one image.

Response:

```json
{
  "reasoning_input": {
    "question_text": "",
    "student_steps": [
      {"step_id": 1, "content": ""}
    ],
    "final_answer": ""
  }
}
```

## Recommended Mode

Use:

```text
ocr_mode = openai_vision
```

This is the high-accuracy mode similar to uploading the image to ChatGPT.

## Setup

```powershell
cd "D:\Research\New folder\ocr_input_understanding"
.\.venv\Scripts\activate
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Open `.env` and set:

```text
OPENAI_API_KEY=your_api_key_here
OPENAI_VISION_MODEL=gpt-5.6
```

## Run

```powershell
uvicorn app:app --reload
```

Open:

```text
http://127.0.0.1:8000/docs
```

Use `POST /extract`.

## Multi-Image Same Question

Use:

```text
POST /extract-pages
```

Use this when the same question or same student answer continues across two or
more photos. Upload the images in the correct order:

```text
image_1 = first page/photo
image_2 = second page/photo
image_3, image_4, image_5 = optional extra pages/photos
```

Recommended settings:

```text
ocr_mode = openai_vision
question_text = optional
```

If both images belong to one question, the response is one combined
`reasoning_input`. If the uploaded images clearly contain separate independent
questions, the response can contain `reasoning_inputs` as an array.

## Marking Scheme Extraction

Use:

```text
POST /extract-marking-scheme
```

Use this when the user uploads a marking-scheme image instead of a student
answer image. This endpoint returns the expected marking steps and marks for
Module 02.

Response:

```json
{
  "marking_scheme": {
    "total_marks": 14,
    "steps": [
      {
        "step_no": 1,
        "description": "Use the remainder theorem to find the remainders for x - 1 and x + 2",
        "marks": 4
      }
    ],
    "final_answer": ""
  }
}
```

Example marking scheme input:

```json
{
  "marking_scheme": {
    "total_marks": 14,
    "steps": [
      {
        "step_no": 1,
        "description": "Use the remainder theorem to find the remainders for x - 1 and x + 2",
        "marks": 4
      },
      {
        "step_no": 2,
        "description": "Correctly calculate P(1) and P(-2) expressions",
        "marks": 3
      },
      {
        "step_no": 3,
        "description": "Use the condition of equal remainders to form an equation",
        "marks": 2
      },
      {
        "step_no": 4,
        "description": "Solve the equation to find the value of a",
        "marks": 2
      },
      {
        "step_no": 5,
        "description": "Substitute a value and calculate the common remainder",
        "marks": 2
      },
      {
        "step_no": 6,
        "description": "State the final answer with a and common remainder",
        "marks": 1
      }
    ]
  }
}
```

Use `/extract` or `/extract-pages` for student answers. Use
`/extract-marking-scheme` for marking schemes.

## Health Check

```text
http://127.0.0.1:8000/
```

The health response shows whether `OPENAI_API_KEY` is configured.

## Result Saving

Every successful extraction is saved to:

```text
outputs/results.json
outputs/<image_id>_result.json
```

## Local Baseline Mode

Use:

```text
ocr_mode = local
```

This runs the local preprocessing, segmentation, EasyOCR, and optional pix2tex
pipeline. It is useful as a baseline for the research report, but the
recommended working system is `openai_vision`.

## Research Framing

The contribution is not creating a new OCR foundation model. The contribution is:

- building an OCR/input-understanding service for handwritten algebra answers
- preprocessing and segmenting realistic answer images
- comparing local OCR with vision-language-model OCR
- returning clean JSON for downstream mathematical reasoning
