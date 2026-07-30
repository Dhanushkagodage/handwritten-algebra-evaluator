from pathlib import Path
import argparse
import csv
import shutil


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_IMAGES_DIR = PROJECT_ROOT / "training" / "data" / "images"
DEFAULT_LABELS_PATH = PROJECT_ROOT / "training" / "data" / "labels.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "training" / "data" / "pix2tex"
VALID_SPLITS = ("train", "val", "test")


def clear_split_folder(split_dir: Path) -> None:
    """Remove exported files for one split without touching source labels/images."""
    images_dir = split_dir / "images"
    equations_path = split_dir / "equations.txt"

    if images_dir.exists():
        for image_file in images_dir.glob("*.png"):
            image_file.unlink()

    images_dir.mkdir(parents=True, exist_ok=True)

    if equations_path.exists():
        equations_path.unlink()


def export_pix2tex_dataset(images_dir: Path, labels_path: Path, output_dir: Path) -> None:
    """Export labeled images and equation text files grouped by train/val/test split."""
    output_dir.mkdir(parents=True, exist_ok=True)

    for split in VALID_SPLITS:
        clear_split_folder(output_dir / split)

    rows_by_split = {split: [] for split in VALID_SPLITS}

    with labels_path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)

        for row in reader:
            filename = row.get("filename", "").strip()
            latex = row.get("latex", "").strip()
            split = row.get("split", "train").strip().lower()

            if not filename or not latex or split not in rows_by_split:
                continue

            rows_by_split[split].append({"filename": filename, "latex": latex})

    for split, rows in rows_by_split.items():
        split_dir = output_dir / split
        split_images_dir = split_dir / "images"
        equations_path = split_dir / "equations.txt"

        with equations_path.open("w", encoding="utf-8", newline="\n") as equations_file:
            for index, row in enumerate(rows, start=1):
                source_image = images_dir / row["filename"]
                target_name = f"{index:06d}.png"
                target_image = split_images_dir / target_name

                shutil.copy2(source_image, target_image)
                equations_file.write(row["latex"] + "\n")

        print(f"Exported {len(rows)} {split} sample(s) to: {split_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export labeled crops into pix2tex-friendly folders.")
    parser.add_argument("--images-dir", default=str(DEFAULT_IMAGES_DIR), help="Folder containing labeled images.")
    parser.add_argument("--labels", default=str(DEFAULT_LABELS_PATH), help="labels.csv path.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="pix2tex dataset output folder.")
    args = parser.parse_args()

    export_pix2tex_dataset(Path(args.images_dir), Path(args.labels), Path(args.output_dir))


if __name__ == "__main__":
    main()

