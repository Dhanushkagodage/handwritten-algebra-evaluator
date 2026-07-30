from pathlib import Path
import os
import re
import shutil
import sys
import time

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from preprocess import preprocess_image
from segment import segment_steps
from ocr_engine import OCREngine
from openai_vision_ocr import extract_reasoning_input_from_images_with_openai, extract_reasoning_input_with_openai
from result_store import save_api_result
from structure_output import build_reasoning_input


load_dotenv(PROJECT_ROOT / ".env")

app = FastAPI(
    title="Handwritten Algebra Extraction API",
    description="Upload a handwritten algebra answer image and receive structured JSON output.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

RAW_DIR = PROJECT_ROOT / "data" / "raw"
PREPROCESSED_DIR = PROJECT_ROOT / "data" / "preprocessed"
REGIONS_DIR = PROJECT_ROOT / "data" / "regions"
OCR_ENGINES = {}


def ensure_dir(path: Path) -> None:
    """Create a folder if it does not already exist."""
    path.mkdir(parents=True, exist_ok=True)


def sanitize_id(value: str) -> str:
    """Convert a user-provided ID into a safe filename-friendly value."""
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "_", value.strip())
    return cleaned.strip("_") or f"API_{int(time.time())}"


def save_uploaded_image(uploaded_file: UploadFile, image_id: str) -> Path:
    """Save an uploaded answer image into data/raw."""
    ensure_dir(RAW_DIR)

    original_suffix = Path(uploaded_file.filename or "").suffix.lower()
    suffix = original_suffix if original_suffix in {".jpg", ".jpeg", ".png"} else ".jpg"
    output_path = RAW_DIR / f"{image_id}{suffix}"

    with output_path.open("wb") as file:
        shutil.copyfileobj(uploaded_file.file, file)

    return output_path


def save_uploaded_images(uploaded_files: list[UploadFile], image_set_id: str) -> list[Path]:
    """Save ordered uploaded answer images into data/raw."""
    saved_paths = []

    for index, uploaded_file in enumerate(uploaded_files, start=1):
        page_id = f"{image_set_id}_page_{index}"
        saved_paths.append(save_uploaded_image(uploaded_file, page_id))

    return saved_paths


def get_ocr_engine(backend: str) -> OCREngine:
    """Create and cache the automatic OCR engine."""
    global OCR_ENGINES

    if backend not in OCR_ENGINES:
        OCR_ENGINES[backend] = OCREngine(backend=backend, manual_fallback=False)

    return OCR_ENGINES[backend]


def make_relative_path(path: str) -> str:
    """Convert a project file path to a clean relative path when possible."""
    path_object = Path(path)

    try:
        return str(path_object.relative_to(PROJECT_ROOT)).replace("\\", "/")
    except ValueError:
        return str(path_object).replace("\\", "/")


@app.get("/")
def health_check() -> dict:
    """Simple health check endpoint."""
    return {
        "status": "ok",
        "message": "OCR Input Understanding API is running.",
        "openai_vision_ready": bool(os.getenv("OPENAI_API_KEY")),
        "openai_vision_model": os.getenv("OPENAI_VISION_MODEL", "gpt-5.6"),
        "local_ocr_available": True,
    }


@app.post("/extract")
def extract_answer(
    image: UploadFile = File(..., description="Handwritten algebra answer image."),
    question_text: str = Form("", description="Optional typed question text for Module 02."),
    ocr_mode: str = Form(
        "openai_vision",
        description="Use 'openai_vision' for ChatGPT-like extraction or 'local' for EasyOCR/pix2tex.",
    ),
    use_math_ocr: bool = Form(False, description="Optional: enable slower pix2tex math OCR."),
) -> dict:
    """
    Extract structured answer data from one uploaded handwritten answer image.

    The user uploads only the image. The API preprocesses it, segments answer
    lines, runs automatic OCR, applies correction rules, and returns JSON.
    """
    safe_image_id = sanitize_id(Path(image.filename or "uploaded_answer").stem)

    try:
        raw_image_path = save_uploaded_image(image, safe_image_id)

        ensure_dir(PREPROCESSED_DIR)
        ensure_dir(REGIONS_DIR)

        if ocr_mode == "openai_vision":
            result = extract_reasoning_input_with_openai(str(raw_image_path), question_text=question_text)
            save_api_result(safe_image_id, result)
            return result

        if ocr_mode != "local":
            raise RuntimeError("Invalid ocr_mode. Use 'openai_vision' or 'local'.")

        preprocessed_path = PREPROCESSED_DIR / f"{safe_image_id}_preprocessed.png"
        preprocess_image(str(raw_image_path), str(preprocessed_path))

        regions = segment_steps(str(preprocessed_path), str(REGIONS_DIR), safe_image_id)
        backend = "hybrid" if use_math_ocr else "easyocr"
        ocr_engine = get_ocr_engine(backend)
        extracted_steps = []

        for region in regions:
            extracted_steps.append(ocr_engine.recognize_step_details(region["image_path"]))
            region["image_path"] = make_relative_path(region["image_path"])

        result = build_reasoning_input(extracted_steps, question_text=question_text)
        save_api_result(safe_image_id, result)
        return result

    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except RuntimeError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error)) from error


@app.post("/extract-pages")
def extract_answer_pages(
    image_1: UploadFile = File(..., description="First answer image."),
    image_2: UploadFile | None = File(None, description="Second answer image, if available."),
    image_3: UploadFile | None = File(None, description="Third answer image, if available."),
    image_4: UploadFile | None = File(None, description="Fourth answer image, if available."),
    image_5: UploadFile | None = File(None, description="Fifth answer image, if available."),
    question_text: str = Form("", description="Optional typed question text for Module 02."),
    ocr_mode: str = Form(
        "openai_vision",
        description="Use 'openai_vision' for best multi-page extraction or 'local' for baseline OCR.",
    ),
    use_math_ocr: bool = Form(False, description="Optional: enable slower pix2tex math OCR for local mode."),
) -> dict:
    """
    Extract structured answer data from multiple ordered images.

    Use this when one question or answer attempt continues across two or more
    photos. The API combines the pages and returns one reasoning input when
    the pages belong to the same question.
    """
    images = [image for image in [image_1, image_2, image_3, image_4, image_5] if image is not None]

    if not images:
        raise HTTPException(status_code=400, detail="Upload at least one image.")

    first_name = Path(images[0].filename or "uploaded_answer").stem
    safe_image_set_id = sanitize_id(f"{first_name}_{int(time.time())}")

    try:
        raw_image_paths = save_uploaded_images(images, safe_image_set_id)

        ensure_dir(PREPROCESSED_DIR)
        ensure_dir(REGIONS_DIR)

        if ocr_mode == "openai_vision":
            result = extract_reasoning_input_from_images_with_openai(
                [str(path) for path in raw_image_paths],
                question_text=question_text,
            )
            save_api_result(safe_image_set_id, result)
            return result

        if ocr_mode != "local":
            raise RuntimeError("Invalid ocr_mode. Use 'openai_vision' or 'local'.")

        backend = "hybrid" if use_math_ocr else "easyocr"
        ocr_engine = get_ocr_engine(backend)
        extracted_steps = []

        for page_index, raw_image_path in enumerate(raw_image_paths, start=1):
            page_id = f"{safe_image_set_id}_page_{page_index}"
            preprocessed_path = PREPROCESSED_DIR / f"{page_id}_preprocessed.png"
            preprocess_image(str(raw_image_path), str(preprocessed_path))

            regions = segment_steps(str(preprocessed_path), str(REGIONS_DIR), page_id)

            for region in regions:
                extracted_steps.append(ocr_engine.recognize_step_details(region["image_path"]))

        result = build_reasoning_input(extracted_steps, question_text=question_text)
        save_api_result(safe_image_set_id, result)
        return result

    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except RuntimeError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error)) from error
