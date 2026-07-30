from pathlib import Path
from datetime import datetime, timezone
import json


PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
RESULTS_PATH = OUTPUTS_DIR / "results.json"


def save_api_result(image_id: str, result: dict) -> None:
    """Save the latest result and append it to outputs/results.json."""
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    record = {
        "image_id": image_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "result": result,
    }

    per_image_path = OUTPUTS_DIR / f"{image_id}_result.json"

    with per_image_path.open("w", encoding="utf-8") as file:
        json.dump(result, file, indent=2, ensure_ascii=False)

    all_results = []

    if RESULTS_PATH.exists():
        try:
            with RESULTS_PATH.open("r", encoding="utf-8") as file:
                loaded = json.load(file)

            if isinstance(loaded, list):
                all_results = loaded
        except json.JSONDecodeError:
            all_results = []

    all_results.append(record)

    with RESULTS_PATH.open("w", encoding="utf-8") as file:
        json.dump(all_results, file, indent=2, ensure_ascii=False)

