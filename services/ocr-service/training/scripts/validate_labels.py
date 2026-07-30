from pathlib import Path
import argparse
import csv


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_IMAGES_DIR = PROJECT_ROOT / "training" / "data" / "images"
DEFAULT_LABELS_PATH = PROJECT_ROOT / "training" / "data" / "labels.csv"
VALID_SPLITS = {"train", "val", "test"}


def validate_labels(images_dir: Path, labels_path: Path) -> bool:
    """Validate that every labeled row has an image, LaTeX value, and valid split."""
    if not labels_path.exists():
        print(f"Missing labels file: {labels_path}")
        return False

    errors = []
    labeled_count = 0
    split_counts = {"train": 0, "val": 0, "test": 0}

    with labels_path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)

        for row_number, row in enumerate(reader, start=2):
            filename = row.get("filename", "").strip()
            latex = row.get("latex", "").strip()
            split = row.get("split", "").strip().lower()

            if not filename:
                errors.append(f"Row {row_number}: missing filename")
                continue

            if not (images_dir / filename).exists():
                errors.append(f"Row {row_number}: image not found: {filename}")

            if not latex:
                errors.append(f"Row {row_number}: missing latex label")
            else:
                labeled_count += 1

            if split not in VALID_SPLITS:
                errors.append(f"Row {row_number}: invalid split '{split}'")
            else:
                split_counts[split] += 1

    if errors:
        print("Validation failed:")
        for error in errors[:30]:
            print(f"- {error}")

        if len(errors) > 30:
            print(f"... and {len(errors) - 30} more error(s)")

        return False

    print("Validation passed.")
    print(f"Labeled image count: {labeled_count}")
    print(f"Split counts: {split_counts}")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate pix2tex label CSV.")
    parser.add_argument("--images-dir", default=str(DEFAULT_IMAGES_DIR), help="Folder containing training images.")
    parser.add_argument("--labels", default=str(DEFAULT_LABELS_PATH), help="labels.csv path.")
    args = parser.parse_args()

    valid = validate_labels(Path(args.images_dir), Path(args.labels))
    raise SystemExit(0 if valid else 1)


if __name__ == "__main__":
    main()

