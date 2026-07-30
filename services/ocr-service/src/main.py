from pathlib import Path
import json
import os

from correction import correct_ocr_text
from ocr_engine import OCREngine
from preprocess import preprocess_image
from segment import segment_steps
from structure_output import build_structured_output


PROJECT_ROOT = Path(__file__).resolve().parent.parent
ANNOTATIONS_PATH = Path("data/annotations.json")
PREPROCESSED_DIR = Path("data/preprocessed")
REGIONS_DIR = Path("data/regions")
RESULTS_PATH = Path("outputs/results.json")


def ensure_dir(path: Path) -> None:
    """Create a folder if it does not already exist."""
    path.mkdir(parents=True, exist_ok=True)


def load_annotations(path: Path) -> list[dict]:
    """Load dataset annotations from JSON."""
    if not path.exists():
        raise FileNotFoundError(f"Annotations file not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        annotations = json.load(file)

    if not isinstance(annotations, list):
        raise ValueError("annotations.json must contain a list of records.")

    return annotations


def save_results(results: list[dict], path: Path) -> None:
    """Save final pipeline results to JSON."""
    ensure_dir(path.parent)

    with path.open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=2, ensure_ascii=False)


def process_annotation(annotation: dict, ocr_engine: OCREngine) -> dict | None:
    """Run the full OCR input-understanding pipeline for one annotation."""
    image_id = annotation.get("image_id", "unknown_image")
    image_path = Path(annotation.get("image_path", ""))

    print()
    print(f"Processing image: {image_id}")

    if not image_path.exists():
        print(f"Skipping {image_id}: image file not found at {image_path}")
        return None

    preprocessed_path = PREPROCESSED_DIR / f"{image_id}_preprocessed.png"

    print("Step 1: Preprocessing image...")
    preprocess_image(str(image_path), str(preprocessed_path))
    print(f"Saved preprocessed image: {preprocessed_path}")

    print("Step 2: Segmenting answer into line/step regions...")
    regions = segment_steps(str(preprocessed_path), str(REGIONS_DIR), image_id)
    print(f"Detected {len(regions)} region(s).")

    if not regions:
        print(f"Skipping {image_id}: no answer regions detected.")
        return build_structured_output(annotation, [], [])

    print("Step 3: Running OCR/manual OCR fallback...")
    recognized_steps = []

    for region in regions:
        raw_text = ocr_engine.recognize_step(region["image_path"])
        corrected_text = correct_ocr_text(raw_text)
        recognized_steps.append(corrected_text)
        print(f"Corrected step {region['step_id']}: {corrected_text}")

    print("Step 4: Building structured JSON object...")
    return build_structured_output(annotation, regions, recognized_steps)


def main() -> None:
    """Run the full pipeline for every image listed in annotations.json."""
    os.chdir(PROJECT_ROOT)
    ensure_dir(PREPROCESSED_DIR)
    ensure_dir(REGIONS_DIR)
    ensure_dir(RESULTS_PATH.parent)

    annotations = load_annotations(ANNOTATIONS_PATH)
    ocr_engine = OCREngine(use_paddle=False, manual_fallback=True)
    results = []

    print("Starting OCR input-understanding pipeline...")
    print(f"Loaded {len(annotations)} annotation record(s).")

    for annotation in annotations:
        result = process_annotation(annotation, ocr_engine)

        if result is not None:
            results.append(result)

    save_results(results, RESULTS_PATH)

    print()
    print(f"Pipeline complete. Saved {len(results)} result(s) to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
