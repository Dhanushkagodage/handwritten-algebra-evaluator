from pathlib import Path

import argparse
import cv2
import matplotlib.pyplot as plt
import numpy as np


def ensure_dir(path: str) -> None:
    """Create the parent folder for a file path if it does not already exist."""
    folder = Path(path).parent
    folder.mkdir(parents=True, exist_ok=True)


def order_points(points: np.ndarray) -> np.ndarray:
    """Order four corner points as top-left, top-right, bottom-right, bottom-left."""
    points = points.reshape(4, 2).astype("float32")
    ordered = np.zeros((4, 2), dtype="float32")

    sums = points.sum(axis=1)
    differences = np.diff(points, axis=1)

    ordered[0] = points[np.argmin(sums)]
    ordered[2] = points[np.argmax(sums)]
    ordered[1] = points[np.argmin(differences)]
    ordered[3] = points[np.argmax(differences)]

    return ordered


def four_point_transform(image: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Apply a top-down perspective transform using four page corners."""
    top_left, top_right, bottom_right, bottom_left = order_points(points)

    width_top = np.linalg.norm(top_right - top_left)
    width_bottom = np.linalg.norm(bottom_right - bottom_left)
    max_width = int(max(width_top, width_bottom))

    height_right = np.linalg.norm(top_right - bottom_right)
    height_left = np.linalg.norm(top_left - bottom_left)
    max_height = int(max(height_right, height_left))

    if max_width < 100 or max_height < 100:
        return image

    destination = np.array(
        [
            [0, 0],
            [max_width - 1, 0],
            [max_width - 1, max_height - 1],
            [0, max_height - 1],
        ],
        dtype="float32",
    )

    matrix = cv2.getPerspectiveTransform(
        np.array([top_left, top_right, bottom_right, bottom_left], dtype="float32"),
        destination,
    )
    return cv2.warpPerspective(image, matrix, (max_width, max_height))


def crop_to_bounding_rect(image: np.ndarray, contour: np.ndarray) -> np.ndarray:
    """Crop an image to a contour bounding rectangle with a small margin."""
    height, width = image.shape[:2]
    x, y, w, h = cv2.boundingRect(contour)

    padding = max(10, int(min(width, height) * 0.02))
    x1 = max(0, x - padding)
    y1 = max(0, y - padding)
    x2 = min(width, x + w + padding)
    y2 = min(height, y + h + padding)

    cropped = image[y1:y2, x1:x2]
    return cropped if cropped.size else image


def detect_and_warp_paper(image: np.ndarray) -> np.ndarray:
    """
    Detect the answer paper inside a phone photo and correct perspective.

    This helps with realistic images that include background, tilt, and partial
    page borders. If no reliable page is found, the original image is returned.
    """
    image = crop_dark_background(image)
    original_height, original_width = image.shape[:2]
    max_dimension = 1000
    scale = max_dimension / max(original_width, original_height)

    if scale < 1.0:
        resized = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    else:
        resized = image.copy()
        scale = 1.0

    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 40, 120)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    edges = cv2.dilate(edges, kernel, iterations=1)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    image_area = resized.shape[0] * resized.shape[1]

    for contour in contours[:8]:
        area = cv2.contourArea(contour)

        if area < image_area * 0.18:
            continue

        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)

        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(approx)
            touches_most_of_image = w > resized.shape[1] * 0.95 and h > resized.shape[0] * 0.95

            if touches_most_of_image:
                continue

            points = approx.reshape(4, 2) / scale
            return four_point_transform(image, points)

    return image


def crop_dark_background(image: np.ndarray) -> np.ndarray:
    """
    Crop away dark non-paper background using row/column brightness projection.

    This is useful for real submissions where the photo includes laptop/table
    background above or beside the answer sheet.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape[:2]

    row_medians = np.median(gray, axis=1)
    row_threshold = _foreground_brightness_threshold(row_medians)

    active_rows = row_medians > row_threshold

    y1, y2 = _largest_active_span(active_rows)

    if y1 is None:
        return image

    padding_y = max(10, int(height * 0.02))

    y1 = max(0, y1 - padding_y)
    y2 = min(height, y2 + padding_y)
    x1 = 0
    x2 = width

    cropped = image[y1:y2, x1:x2]

    if cropped.size == 0:
        return image

    crop_area = cropped.shape[0] * cropped.shape[1]
    image_area = height * width

    if crop_area < image_area * 0.25:
        return image

    return cropped


def _foreground_brightness_threshold(values: np.ndarray) -> float:
    """Choose a brightness threshold that keeps shadowed paper but removes dark background."""
    dark_value = float(np.percentile(values, 10))
    bright_value = float(np.percentile(values, 90))
    return max(60.0, dark_value + 0.35 * (bright_value - dark_value))


def _largest_active_span(active_values: np.ndarray) -> tuple[int | None, int | None]:
    """Find the largest continuous active span in a boolean vector."""
    best_start = None
    best_end = None
    current_start = None

    for index, is_active in enumerate(active_values):
        if is_active and current_start is None:
            current_start = index
        elif not is_active and current_start is not None:
            current_end = index

            if best_start is None or current_end - current_start > best_end - best_start:
                best_start = current_start
                best_end = current_end

            current_start = None

    if current_start is not None:
        current_end = len(active_values)

        if best_start is None or current_end - current_start > best_end - best_start:
            best_start = current_start
            best_end = current_end

    return best_start, best_end


def crop_to_largest_light_region(image: np.ndarray) -> np.ndarray:
    """
    Fallback page crop using the largest bright/low-saturation region.

    This removes dark laptop/table background when page-corner detection is not
    reliable enough for perspective correction.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]

    paper_mask = cv2.inRange(value, 80, 255)
    low_saturation_mask = cv2.inRange(saturation, 0, 130)
    paper_mask = cv2.bitwise_and(paper_mask, low_saturation_mask)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    paper_mask = cv2.morphologyEx(paper_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    paper_mask = cv2.morphologyEx(paper_mask, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(paper_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return image

    largest = max(contours, key=cv2.contourArea)
    image_area = image.shape[0] * image.shape[1]

    if cv2.contourArea(largest) < image_area * 0.20:
        return image

    return crop_to_bounding_rect(image, largest)


def normalize_illumination(gray_image: np.ndarray) -> np.ndarray:
    """
    Reduce shadows and uneven lighting before thresholding.

    A blurred background estimate is divided out from the grayscale image, then
    CLAHE boosts local contrast for faint handwriting.
    """
    background = cv2.GaussianBlur(gray_image, (0, 0), sigmaX=25, sigmaY=25)
    normalized = cv2.divide(gray_image, background, scale=255)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(normalized)


def correct_skew(image: np.ndarray) -> np.ndarray:
    """
    Try to straighten a scanned or photographed answer image.

    This is a light skew correction step. If the angle cannot be estimated,
    the original image is returned.
    """
    coordinates = np.column_stack(np.where(image < 255))

    if len(coordinates) == 0:
        return image

    angle = cv2.minAreaRect(coordinates)[-1]

    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    if abs(angle) < 0.5:
        return image

    height, width = image.shape[:2]
    center = (width // 2, height // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

    corrected = cv2.warpAffine(
        image,
        rotation_matrix,
        (width, height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )

    return corrected


def show_debug_image(title: str, image: np.ndarray) -> None:
    """Show an image while developing or debugging the preprocessing step."""
    plt.figure(figsize=(10, 6))
    plt.imshow(image, cmap="gray")
    plt.title(title)
    plt.axis("off")
    plt.show()


def preprocess_image(input_path: str, output_path: str) -> str:
    """
    Clean a handwritten algebra answer image and save the preprocessed version.

    The saved image is a black-and-white thresholded image that is easier to
    segment and pass to OCR in the next phases.
    """
    input_file = Path(input_path)

    if not input_file.exists():
        raise FileNotFoundError(f"Input image not found: {input_path}")

    image = cv2.imread(str(input_file))

    if image is None:
        raise ValueError(f"Could not read image file: {input_path}")

    paper_image = detect_and_warp_paper(image)
    gray_image = cv2.cvtColor(paper_image, cv2.COLOR_BGR2GRAY)
    normalized_image = normalize_illumination(gray_image)
    denoised_image = cv2.medianBlur(normalized_image, 3)

    thresholded_image = cv2.adaptiveThreshold(
        denoised_image,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        15,
    )

    corrected_image = correct_skew(thresholded_image)

    ensure_dir(output_path)
    saved = cv2.imwrite(output_path, corrected_image)

    if not saved:
        raise IOError(f"Could not save preprocessed image: {output_path}")

    return output_path


def build_output_path(input_path: str) -> str:
    """Create a default preprocessed output path from an input image path."""
    input_file = Path(input_path)
    output_name = f"{input_file.stem}_preprocessed.png"
    return str(Path("data/preprocessed") / output_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess a handwritten answer image.")
    parser.add_argument(
        "--input",
        default="data/raw/Q01_FC_W01.jpg",
        help="Path to the raw handwritten answer image.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Path where the preprocessed image should be saved.",
    )
    args = parser.parse_args()

    output_path = args.output or build_output_path(args.input)
    result_path = preprocess_image(args.input, output_path)
    print(f"Preprocessed image saved to: {result_path}")
