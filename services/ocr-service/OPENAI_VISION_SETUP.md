# OpenAI Vision OCR Setup

This backend uses an OpenAI vision-capable model to extract the handwritten
answer directly into the Module 01 JSON shape.

It is closer to the result you see when uploading the image to ChatGPT.

## 1. Install Requirements

```powershell
cd "D:\Research\New folder\ocr_input_understanding"
.\.venv\Scripts\activate
python -m pip install -r requirements.txt
```

## 2. Set API Key

PowerShell:

```powershell
$env:OPENAI_API_KEY="your_api_key_here"
```

Optional model override:

```powershell
$env:OPENAI_VISION_MODEL="gpt-5.6"
```

## 3. Run API

```powershell
uvicorn app:app --reload
```

Open:

```text
http://127.0.0.1:8000/docs
```

Use:

```text
POST /extract
ocr_mode = openai_vision
```

## Output

The response is only:

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

## Notes

- This uses an external API, so it requires internet access and API credits.
- Images are sent to OpenAI for processing.
- For a fully offline/local research prototype, use `ocr_mode = local`.

