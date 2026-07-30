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
