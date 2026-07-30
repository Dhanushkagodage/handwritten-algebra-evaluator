from pathlib import Path
import argparse

import cv2
import numpy as np


def ensure_dir(path: str) -> None:
    """Create a folder if it does not already exist."""
    Path(path).mkdir(parents=True, exist_ok=True)


def _foreground_mask(gray_image: np.ndarray) -> np.ndarray:
    """
    Convert a grayscale image into a foreground mask.

    In the returned mask, handwriting/text pixels are white and the background
    is black. This makes line detection easier.
    """
    blurred = cv2.GaussianBlur(gray_image, (3, 3), 0)
    _, mask = cv2.threshold(
        blurred,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
    )
    return _clean_foreground_mask(mask)


def _clean_foreground_mask(mask: np.ndarray) -> np.ndarray:
    """
    Remove notebook ruling, borders, and tiny specks from the foreground mask.

    Phone photos of ruled pages often contain many horizontal notebook lines.
    If we keep those lines, multiple answer rows get connected together and OCR
    receives one huge crop. This cleanup keeps handwriting-like components and
    removes long thin horizontal components.
    """
    image_height, image_width = mask.shape[:2]

    horizontal_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (max(40, image_width // 3), 1),
    )
    horizontal_lines = cv2.morphologyEx(mask, cv2.MORPH_OPEN, horizontal_kernel)
    cleaned = cv2.subtract(mask, horizontal_lines)

    cleaned = _remove_ruled_page_lines(cleaned)
    cleaned = _remove_ruled_page_rows(cleaned)

    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(cleaned, connectivity=8)
    filtered = np.zeros_like(cleaned)

    min_area = max(8, int(image_width * image_height * 0.000005))
    max_thin_line_height = max(8, image_height // 150)

    for label in range(1, component_count):
        x = stats[label, cv2.CC_STAT_LEFT]
        y = stats[label, cv2.CC_STAT_TOP]
        w = stats[label, cv2.CC_STAT_WIDTH]
        h = stats[label, cv2.CC_STAT_HEIGHT]
        area = stats[label, cv2.CC_STAT_AREA]

        if area < min_area:
            continue

        is_long_thin_line = w > image_width * 0.45 and h <= max_thin_line_height
        is_page_edge = w > image_width * 0.70 and h <= image_height * 0.03
        is_tall_border = h > image_height * 0.20 and w <= image_width * 0.02

        if is_long_thin_line or is_page_edge or is_tall_border:
            continue

        filtered[labels == label] = 255

    return filtered


def _remove_ruled_page_lines(mask: np.ndarray) -> np.ndarray:
    """
    Remove long near-horizontal ruled lines, including slightly tilted ones.

    Morphological horizontal kernels miss notebook lines when the phone photo is
    angled. Hough line detection catches those slanted line segments.
    """
    image_height, image_width = mask.shape[:2]
    cleaned = mask.copy()
    lines = cv2.HoughLinesP(
        cleaned,
        rho=1,
        theta=np.pi / 180,
        threshold=max(35, image_width // 18),
        minLineLength=max(80, image_width // 3),
        maxLineGap=max(15, image_width // 25),
    )

    if lines is None:
        return cleaned

    for line in lines:
        x1, y1, x2, y2 = np.array(line).reshape(-1)[:4]
        dx = x2 - x1
        dy = y2 - y1

        if dx == 0:
            angle = 90.0
        else:
            angle = abs(np.degrees(np.arctan2(dy, dx)))

        length = np.hypot(dx, dy)
        is_ruled_line = angle <= 12 and length > image_width * 0.25
        is_vertical_margin = angle >= 78 and length > image_height * 0.20

        if is_ruled_line or is_vertical_margin:
            cv2.line(cleaned, (x1, y1), (x2, y2), 0, thickness=5)

    return cleaned


def _remove_ruled_page_rows(mask: np.ndarray) -> np.ndarray:
    """
    Remove broken/dashed notebook ruling rows.

    Ruled lines in phone photos often become dashed after thresholding. They do
    not form one connected horizontal line, but they still create many ink
    pixels on the same row. We detect those high-ink rows and erase a thin band
    around them.
    """
    image_height, image_width = mask.shape[:2]
    cleaned = mask.copy()
    rule_rows = []

    for row_index in range(image_height):
        columns = np.where(cleaned[row_index, :] > 0)[0]

        if len(columns) == 0:
            continue

        row_span = int(columns[-1] - columns[0] + 1)
        ink_count = len(columns)
        is_wide_dashed_rule = row_span > image_width * 0.72 and ink_count > image_width * 0.045

        if is_wide_dashed_rule:
            rule_rows.append(row_index)

    for row in rule_rows:
        y1 = max(0, row - 2)
        y2 = min(image_height, row + 3)
        cleaned[y1:y2, :] = 0

    return cleaned


def _pad_box(box: tuple[int, int, int, int], image_width: int, image_height: int, padding: int) -> list[int]:
    """Add safe padding around a bounding box without going outside the image."""
    x, y, w, h = box
    x1 = max(0, x - padding)
    y1 = max(0, y - padding)
    x2 = min(image_width, x + w + padding)
    y2 = min(image_height, y + h + padding)
    return [x1, y1, x2 - x1, y2 - y1]


def _merge_close_boxes(boxes: list[list[int]], max_gap: int = 12) -> list[list[int]]:
    """
    Merge boxes that belong to the same handwritten line.

    Sometimes one answer step is detected as multiple parts, especially when
    there are gaps around operators or brackets. This function joins boxes that
    overlap vertically or are very close to each other.
    """
    if not boxes:
        return []

    sorted_boxes = sorted(boxes, key=lambda box: box[1])
    merged = [sorted_boxes[0]]

    for box in sorted_boxes[1:]:
        current = merged[-1]
        current_top = current[1]
        current_bottom = current[1] + current[3]
        box_top = box[1]
        box_bottom = box[1] + box[3]

        overlaps = box_top <= current_bottom and box_bottom >= current_top
        close = abs(box_top - current_bottom) <= max_gap

        if overlaps or close:
            x1 = min(current[0], box[0])
            y1 = min(current[1], box[1])
            x2 = max(current[0] + current[2], box[0] + box[2])
            y2 = max(current[1] + current[3], box[1] + box[3])
            merged[-1] = [x1, y1, x2 - x1, y2 - y1]
        else:
            merged.append(box)

    return merged


def _detect_steps_by_contours(mask: np.ndarray) -> list[list[int]]:
    """
    Detect line/step regions using contours.

    Horizontal dilation joins nearby symbols in the same line, then contours
    are used to find bounding boxes around those joined regions.
    """
    image_height, image_width = mask.shape[:2]
    kernel_width = max(25, image_width // 25)
    kernel_height = max(3, image_height // 180)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_width, kernel_height))
    dilated = cv2.dilate(mask, kernel, iterations=1)

    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []

    min_width = max(20, int(image_width * 0.04))
    min_height = max(8, int(image_height * 0.01))

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)

        if w < min_width or h < min_height:
            continue

        padded_box = _pad_box((x, y, w, h), image_width, image_height, padding=8)
        boxes.append(padded_box)

    return _merge_close_boxes(boxes)


def _detect_steps_by_projection(mask: np.ndarray) -> list[list[int]]:
    """
    Fallback line detection using horizontal projection.

    The idea is simple: rows containing handwriting have more dark/ink pixels.
    Consecutive active rows are grouped into line regions.
    """
    image_height, image_width = mask.shape[:2]
    row_ink_counts = np.sum(mask > 0, axis=1)
    active_threshold = max(4, int(image_width * 0.01))
    active_rows = row_ink_counts > active_threshold

    max_gap = max(8, image_height // 140)
    min_line_height = max(8, image_height // 160)

    spans = []
    start = None
    last_active = None

    for row_index, is_active in enumerate(active_rows):
        if is_active:
            if start is None:
                start = row_index
            last_active = row_index
        elif start is not None and row_index - last_active > max_gap:
            spans.append((start, last_active))
            start = None
            last_active = None

    if start is not None:
        spans.append((start, last_active))

    boxes = []

    for y1, y2 in spans:
        if y2 - y1 + 1 < min_line_height:
            continue

        line_mask = mask[y1 : y2 + 1, :]
        columns = np.where(np.sum(line_mask > 0, axis=0) > 0)[0]

        if len(columns) == 0:
            continue

        x1 = int(columns[0])
        x2 = int(columns[-1])
        box = _pad_box((x1, y1, x2 - x1 + 1, y2 - y1 + 1), image_width, image_height, padding=10)
        boxes.append(box)

    return _merge_close_boxes(boxes)


def _split_tall_boxes_by_projection(boxes: list[list[int]], mask: np.ndarray) -> list[list[int]]:
    """Split very tall boxes into smaller answer lines using row projection."""
    image_height, image_width = mask.shape[:2]
    split_boxes = []
    tall_box_limit = max(110, int(image_height * 0.10))

    for x, y, w, h in boxes:
        if h <= tall_box_limit:
            split_boxes.append([x, y, w, h])
            continue

        sub_mask = mask[y : y + h, x : x + w]
        sub_boxes = _detect_steps_by_projection(sub_mask)

        if len(sub_boxes) <= 1:
            sub_boxes = _force_split_tall_box(sub_mask)

        if len(sub_boxes) <= 1:
            split_boxes.append([x, y, w, h])
            continue

        for sub_x, sub_y, sub_w, sub_h in sub_boxes:
            split_boxes.append([x + sub_x, y + sub_y, sub_w, sub_h])

    return sorted(split_boxes, key=lambda box: box[1])


def _force_split_tall_box(mask: np.ndarray) -> list[list[int]]:
    """
    Force a tall answer area into rows using valleys in horizontal ink density.

    This is a final fallback for ruled pages where small noise keeps the whole
    answer body connected.
    """
    image_height, image_width = mask.shape[:2]
    row_ink_counts = np.sum(mask > 0, axis=1).astype(np.float32)

    if image_height < 80:
        return []

    smooth_kernel = max(9, image_height // 45)
    if smooth_kernel % 2 == 0:
        smooth_kernel += 1

    smoothed = cv2.GaussianBlur(row_ink_counts.reshape(-1, 1), (1, smooth_kernel), 0).flatten()
    active_threshold = max(3, float(np.percentile(smoothed, 65)) * 0.35)
    active_rows = smoothed > active_threshold

    # Close tiny gaps inside the same handwriting row, but keep real blank gaps.
    max_gap = max(5, image_height // 100)
    spans = []
    start = None
    last_active = None

    for row_index, is_active in enumerate(active_rows):
        if is_active:
            if start is None:
                start = row_index
            last_active = row_index
        elif start is not None and row_index - last_active > max_gap:
            spans.append((start, last_active))
            start = None
            last_active = None

    if start is not None:
        spans.append((start, last_active))

    boxes = []
    min_height = max(8, image_height // 80)

    for y1, y2 in spans:
        if y2 - y1 + 1 < min_height:
            continue

        line_mask = mask[y1 : y2 + 1, :]
        columns = np.where(np.sum(line_mask > 0, axis=0) > 0)[0]

        if len(columns) == 0:
            continue

        x1 = int(columns[0])
        x2 = int(columns[-1])
        boxes.append(_pad_box((x1, y1, x2 - x1 + 1, y2 - y1 + 1), image_width, image_height, 8))

    return boxes


def _filter_step_boxes(boxes: list[list[int]], mask: np.ndarray) -> list[list[int]]:
    """Remove boxes that are probably noise instead of answer steps."""
    image_height, image_width = mask.shape[:2]
    filtered = []

    for x, y, w, h in boxes:
        region = mask[y : y + h, x : x + w]
        ink_pixels = int(np.sum(region > 0))
        ink_rows = int(np.sum(np.sum(region > 0, axis=1) > 2))
        ink_density = ink_pixels / max(1, w * h)
        component_count, _, component_stats, _ = cv2.connectedComponentsWithStats(region, connectivity=8)
        component_heights = [
            component_stats[label, cv2.CC_STAT_HEIGHT]
            for label in range(1, component_count)
            if component_stats[label, cv2.CC_STAT_AREA] >= 3
        ]
        max_component_height = max(component_heights, default=0)

        if ink_pixels < 18:
            continue

        if w < image_width * 0.04 or h < image_height * 0.006:
            continue

        if w > image_width * 0.80 and h < image_height * 0.025:
            continue

        is_top_right_background = y < image_height * 0.18 and x > image_width * 0.50
        is_thin_rule_fragment = ink_rows < max(6, int(h * 0.22))
        is_top_page_edge = y < image_height * 0.25 and w > image_width * 0.80 and ink_density < 0.06
        has_handwriting_component = max_component_height >= max(8, int(h * 0.18))

        if is_top_right_background or is_thin_rule_fragment or is_top_page_edge or not has_handwriting_component:
            continue

        filtered.append([x, y, w, h])

    return filtered


def _drop_leading_photo_noise(boxes: list[list[int]], mask: np.ndarray) -> list[list[int]]:
    """
    Drop accidental top-of-photo detections before the answer page starts.

    Cropped phone photos may include table/laptop/background above the paper.
    Those artifacts usually appear before the first wide handwriting line.
    """
    if len(boxes) <= 1:
        return boxes

    image_height, image_width = mask.shape[:2]
    sorted_boxes = sorted(boxes, key=lambda box: box[1])

    boxes_below_top_noise = [box for box in sorted_boxes if box[1] > image_height * 0.18]

    if len(boxes_below_top_noise) >= 3:
        sorted_boxes = [
            box
            for box in sorted_boxes
            if box[1] > image_height * 0.15 or box[2] > image_width * 0.45
        ]

        if not sorted_boxes:
            return boxes_below_top_noise

    first_answer_index = 0

    for index, (x, y, w, h) in enumerate(sorted_boxes):
        region = mask[y : y + h, x : x + w]
        ink_pixels = int(np.sum(region > 0))
        is_strong_text_line = w > image_width * 0.22 and ink_pixels > 45

        if is_strong_text_line:
            first_answer_index = index
            break

    if first_answer_index == 0:
        return sorted_boxes

    first_answer_y = sorted_boxes[first_answer_index][1]
    has_large_top_gap = first_answer_y > image_height * 0.12

    if has_large_top_gap:
        return sorted_boxes[first_answer_index:]

    return sorted_boxes


def segment_steps(preprocessed_image_path: str, output_dir: str, image_id: str) -> list[dict]:
    """
    Crop handwritten answer lines/steps from a preprocessed image.

    Each cropped step image is saved to output_dir and metadata for every step
    is returned as a list of dictionaries.
    """
    input_file = Path(preprocessed_image_path)

    if not input_file.exists():
        raise FileNotFoundError(f"Preprocessed image not found: {preprocessed_image_path}")

    image = cv2.imread(str(input_file), cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise ValueError(f"Could not read image file: {preprocessed_image_path}")

    ensure_dir(output_dir)

    for old_region in Path(output_dir).glob(f"{image_id}_step_*.png"):
        old_region.unlink()

    mask = _foreground_mask(image)
    boxes = _detect_steps_by_projection(mask)

    if len(boxes) == 0:
        boxes = _detect_steps_by_contours(mask)

    boxes = _split_tall_boxes_by_projection(boxes, mask)
    boxes = _filter_step_boxes(boxes, mask)
    boxes = _drop_leading_photo_noise(boxes, mask)

    boxes = sorted(boxes, key=lambda box: box[1])
    regions = []

    for index, box in enumerate(boxes, start=1):
        x, y, w, h = box
        # Save the cleaned mask crop, not the original threshold crop. This
        # removes ruled notebook lines from the OCR input while keeping the
        # handwriting strokes.
        cropped_step = 255 - mask[y : y + h, x : x + w]
        region_path = Path(output_dir) / f"{image_id}_step_{index}.png"

        saved = cv2.imwrite(str(region_path), cropped_step)

        if not saved:
            raise IOError(f"Could not save cropped region: {region_path}")

        regions.append(
            {
                "step_id": index,
                "bbox": [int(x), int(y), int(w), int(h)],
                "image_path": str(region_path).replace("\\", "/"),
            }
        )

    return regions


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Segment a preprocessed answer image into step regions.")
    parser.add_argument(
        "--input",
        required=True,
        help="Path to the preprocessed image.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/regions",
        help="Folder where cropped step images should be saved.",
    )
    parser.add_argument(
        "--image-id",
        required=True,
        help="Image ID used when naming cropped step files.",
    )
    args = parser.parse_args()

    detected_regions = segment_steps(args.input, args.output_dir, args.image_id)

    print(f"Detected {len(detected_regions)} step region(s).")
    for region in detected_regions:
        print(region)
