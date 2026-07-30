# PaddleOCR Setup on Windows

Use a 64-bit Python version that PaddlePaddle supports. Recommended:

```text
Python 3.11 64-bit
```

Avoid using an unsupported or too-new Python version such as Python 3.14 if
PaddlePaddle wheels are not available for it.

## 1. Check Your Python Version

```powershell
python --version
python -c "import platform, struct; print(platform.python_version()); print(platform.architecture()[0]); print(struct.calcsize('P') * 8)"
```

You should see a 64-bit Python version such as:

```text
3.11.x
64bit
64
```

## 2. Create a Clean Virtual Environment

From the project folder:

```powershell
cd "D:\Research\New folder\ocr_input_understanding"
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip setuptools wheel
```

## 3. Install Basic Requirements

```powershell
python -m pip install -r requirements.txt
```

## 4. Install PaddleOCR Requirements

```powershell
python -m pip install -r requirements-ocr.txt
```

If `paddlepaddle` gives `No matching distribution found`, your Python version
or architecture is not compatible. Install Python 3.11 64-bit and repeat the
virtual environment steps.

## 5. Verify Paddle Installation

```powershell
python -c "import paddle; paddle.utils.run_check()"
python -c "from paddleocr import PaddleOCR; print('PaddleOCR ready')"
```

## 6. Run the API

```powershell
uvicorn app:app --reload
```

Open:

```text
http://127.0.0.1:8000/docs
```

