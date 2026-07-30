# EasyOCR Setup

EasyOCR is the recommended OCR backend for the current prototype because it is
easier to install than PaddleOCR and can run on CPU.

## 1. Activate Your Virtual Environment

```powershell
cd "D:\Research\New folder\ocr_input_understanding"
.\.venv\Scripts\activate
```

If your virtual environment has problems, create a clean one:

```powershell
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip setuptools wheel
```

## 2. Install Requirements

```powershell
python -m pip install -r requirements.txt
python -m pip install -r requirements-ocr.txt
```

## 3. Verify EasyOCR

```powershell
python -c "import easyocr; print('EasyOCR ready')"
```

## 4. Run the API

```powershell
uvicorn app:app --reload
```

Open:

```text
http://127.0.0.1:8000/docs
```

Upload a handwritten answer image using `POST /extract`.

## Note About pix2tex

pix2tex / LaTeX-OCR can be added later if the project needs LaTeX-style math
output. For now, EasyOCR is simpler for completing the upload-to-JSON pipeline.

