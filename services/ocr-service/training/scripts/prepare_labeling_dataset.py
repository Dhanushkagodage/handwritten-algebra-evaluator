from pathlib import Path
import argparse
import csv
import shutil


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE_DIR = PROJECT_ROOT / "data" / "regions"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "training" / "data" / "images"
DEFAULT_LABELS_PATH = PROJECT_ROOT / "training" / "data" / "labels.csv"


def read_existing_labels(labels_path: Path) -> dict[str, dict]:
    """Keep existing LaTeX labels if the CSV already exists."""
    if not labels_path.exists():
        return {}

    with labels_path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        return {row["filename"]: row for row in reader if row.get("filename")}


def prepare_labeling_dataset(
    source_dir: Path,
    output_dir: Path,
    labels_path: Path,
    include_debug: bool = False,
) -> None:
    """Copy cropped step images into a stable training image folder and create labels.csv."""
    output_dir.mkdir(parents=True, exist_ok=True)
    labels_path.parent.mkdir(parents=True, exist_ok=True)

    existing_labels = read_existing_labels(labels_path)
    source_images = sorted(source_dir.glob("*.png"))

    if not include_debug:
        source_images = [image for image in source_images if "debug" not in image.stem.lower()]

    rows = []

    for index, source_image in enumerate(source_images, start=1):
        target_name = f"sample_{index:05d}.png"
        target_path = output_dir / target_name

        if not target_path.exists():
            shutil.copy2(source_image, target_path)

        previous = existing_labels.get(target_name, {})
        rows.append(
            {
                "filename": target_name,
                "latex": previous.get("latex", ""),
                "split": previous.get("split", "train"),
                "source_image": str(source_image.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            }
        )

    with labels_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["filename", "latex", "split", "source_image"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Copied {len(rows)} image(s) to: {output_dir}")
    print(f"Label file written to: {labels_path}")
    print("Fill the latex column before exporting the pix2tex dataset.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare cropped images for manual LaTeX labeling.")
    parser.add_argument("--source-dir", default=str(DEFAULT_SOURCE_DIR), help="Folder containing cropped step images.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Training image output folder.")
    parser.add_argument("--labels", default=str(DEFAULT_LABELS_PATH), help="labels.csv output path.")
    parser.add_argument(
        "--include-debug",
        action="store_true",
        help="Include files with 'debug' in the name. Disabled by default.",
    )
    args = parser.parse_args()

    prepare_labeling_dataset(
        Path(args.source_dir),
        Path(args.output_dir),
        Path(args.labels),
        include_debug=args.include_debug,
    )


if __name__ == "__main__":
    main()
