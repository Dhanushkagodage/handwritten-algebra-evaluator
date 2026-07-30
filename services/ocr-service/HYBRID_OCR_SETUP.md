# Hybrid OCR Setup

This project now uses hybrid OCR:

- EasyOCR for ordinary handwritten text
- pix2tex / LaTeX-OCR for handwritten math expressions

## Install

```powershell
cd "D:\Research\New folder\ocr_input_understanding"
.\.venv\Scripts\activate
python -m pip install -r requirements.txt
python -m pip install -r requirements-ocr.txt
```

The first run may download EasyOCR and pix2tex model files.

## Verify

```powershell
python -c "import easyocr; print('EasyOCR ready')"
python -c "from pix2tex.cli import LatexOCR; print('pix2tex ready')"
```

## Run API

```powershell
uvicorn app:app --reload
```

Open:

```text
http://127.0.0.1:8000/docs
```

Upload a handwritten answer image using `POST /extract`.

## Expected Behavior

Each segmented row is processed like this:

```text
row image -> EasyOCR text hint
          -> if math-heavy, pix2tex
          -> plain expression + raw_latex in JSON
```

If pix2tex fails for a row, the system falls back to EasyOCR for that row.

