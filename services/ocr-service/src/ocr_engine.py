from pathlib import Path
import argparse
import importlib.util
import re

from correction import correct_ocr_text


class OCREngine:
    """
    OCR wrapper for recognizing text from cropped answer step images.

    The engine supports:
    - EasyOCR for ordinary handwritten text
    - pix2tex / LaTeX-OCR for math-heavy handwritten expressions
    - PaddleOCR as an optional backend
    - manual OCR for terminal testing
    """

    def __init__(
        self,
        use_paddle: bool = False,
        manual_fallback: bool = True,
        backend: str = "manual",
    ):
        if use_paddle:
            backend = "paddle"

        self.manual_fallback = manual_fallback
        self.backend_name = "manual"
        self.backend = backend
        self.paddle_ocr = None
        self.easyocr_reader = None
        self.pix2tex_model = None

        if self.backend == "auto":
            self._load_auto_backend()
        elif self.backend == "paddle":
            self._load_paddleocr()
        elif self.backend == "easyocr":
            self._load_easyocr()
        elif self.backend == "pix2tex":
            self._load_pix2tex()
        elif self.backend == "hybrid":
            self._load_hybrid_backends()

    def _load_auto_backend(self) -> None:
        """Load the first available automatic OCR backend."""
        if importlib.util.find_spec("paddleocr") is not None:
            self._load_paddleocr()
            return

        if importlib.util.find_spec("pix2tex") is not None and importlib.util.find_spec("easyocr") is not None:
            self._load_hybrid_backends()
            return

        if importlib.util.find_spec("easyocr") is not None:
            self._load_easyocr()
            return

        if self.manual_fallback:
            print("No automatic OCR backend found. Using manual OCR fallback.")
            return

        raise RuntimeError(
            "No automatic OCR backend is installed. Run: "
            "python -m pip install -r requirements-ocr.txt"
        )

    def _load_paddleocr(self) -> None:
        """
        Load PaddleOCR only when it is installed and requested.

        PaddleOCR is imported lazily so the rest of the project can still run
        in environments where the OCR model is not installed.
        """
        if importlib.util.find_spec("paddleocr") is None:
            if self.manual_fallback:
                print("PaddleOCR is not installed. Using manual OCR fallback.")
                return

            raise RuntimeError(
                "PaddleOCR is not installed. Install it with: "
                "python -m pip install -r requirements-ocr.txt"
            )

        from paddleocr import PaddleOCR

        try:
            self.paddle_ocr = PaddleOCR(use_angle_cls=True, lang="en", show_log=False)
        except TypeError:
            self.paddle_ocr = PaddleOCR(use_angle_cls=True, lang="en")

        self.backend_name = "paddleocr"

    def _load_easyocr(self) -> None:
        """
        Load EasyOCR as a CPU-friendly automatic OCR backend.

        EasyOCR is not perfect for handwritten algebra, but it gives the API a
        working automatic OCR path without requiring PaddlePaddle.
        """
        if importlib.util.find_spec("easyocr") is None:
            if self.manual_fallback:
                print("EasyOCR is not installed. Using manual OCR fallback.")
                return

            raise RuntimeError(
                "EasyOCR is not installed. Install it with: "
                "python -m pip install -r requirements-ocr.txt"
            )

        import easyocr

        self.easyocr_reader = easyocr.Reader(["en"], gpu=False)
        self.backend_name = "easyocr"

    def _load_pix2tex(self) -> None:
        """
        Load pix2tex / LaTeX-OCR for handwritten math expressions.

        pix2tex can be slow to load, so it is initialized lazily only when this
        backend is selected.
        """
        if importlib.util.find_spec("pix2tex") is None:
            if self.manual_fallback:
                print("pix2tex is not installed. Using manual OCR fallback.")
                return

            raise RuntimeError(
                "pix2tex is not installed. Install it with: "
                "python -m pip install -r requirements-ocr.txt"
            )

        from pix2tex.cli import LatexOCR

        self.pix2tex_model = LatexOCR()
        self.backend_name = "pix2tex"

    def _load_hybrid_backends(self) -> None:
        """Load EasyOCR and pix2tex for hybrid text/math OCR."""
        missing_packages = []

        if importlib.util.find_spec("easyocr") is None:
            missing_packages.append("easyocr")

        if importlib.util.find_spec("pix2tex") is None:
            missing_packages.append("pix2tex")

        if missing_packages:
            if self.manual_fallback:
                print(f"Missing OCR packages: {', '.join(missing_packages)}. Using manual OCR fallback.")
                return

            raise RuntimeError(
                "Missing OCR packages for hybrid OCR: "
                f"{', '.join(missing_packages)}. Run: python -m pip install -r requirements-ocr.txt"
            )

        self._load_easyocr()
        self._load_pix2tex()
        self.backend_name = "hybrid_easyocr_pix2tex"

    def recognize_step(self, image_path: str) -> str:
        """
        Recognize one cropped answer step image.

        Automatic OCR is used first when available. Manual OCR is used only
        when automatic OCR is unavailable and manual_fallback=True.
        """
        result = self.recognize_step_details(image_path)
        return result["text"]

    def recognize_step_details(self, image_path: str) -> dict:
        """
        Recognize one cropped answer step and return text plus OCR metadata.
        """
        image_file = Path(image_path)

        if not image_file.exists():
            raise FileNotFoundError(f"Step image not found: {image_path}")

        if self.backend == "hybrid" and self.easyocr_reader is not None and self.pix2tex_model is not None:
            return self._recognize_with_hybrid(image_file)

        if self.paddle_ocr is not None:
            text = self._recognize_with_paddle(image_file)
            return {"text": text, "backend": "paddleocr", "raw_latex": "", "easyocr_text": ""}

        if self.easyocr_reader is not None:
            text = self._recognize_with_easyocr(image_file)
            return {"text": text, "backend": "easyocr", "raw_latex": "", "easyocr_text": text}

        if self.pix2tex_model is not None:
            latex = self._recognize_latex_with_pix2tex(image_file)
            text = latex_to_plain_text(latex)
            return {"text": text, "backend": "pix2tex", "raw_latex": latex, "easyocr_text": ""}

        if not self.manual_fallback:
            raise RuntimeError("Manual fallback is disabled and no OCR backend is available.")

        text = self._manual_recognize(image_file)
        return {"text": text, "backend": "manual", "raw_latex": "", "easyocr_text": ""}

    def recognize_steps(self, regions: list[dict]) -> list[str]:
        """
        Recognize multiple cropped regions and return their text in order.
        """
        recognized_text = []

        for region in regions:
            image_path = region.get("image_path")

            if not image_path:
                raise ValueError(f"Region is missing image_path: {region}")

            text = self.recognize_step(image_path)
            recognized_text.append(text)

        return recognized_text

    def _recognize_with_paddle(self, image_file: Path) -> str:
        """Recognize text from an image using PaddleOCR."""
        try:
            result = self.paddle_ocr.ocr(str(image_file), cls=True)
        except TypeError:
            result = self.paddle_ocr.ocr(str(image_file))

        text_parts = self._extract_text_parts(result)
        return correct_ocr_text(" ".join(text_parts))

    def _recognize_with_easyocr(self, image_file: Path) -> str:
        """Recognize text from an image using EasyOCR."""
        result = self.easyocr_reader.readtext(str(image_file), detail=1, paragraph=False)

        text_parts = []

        for item in result:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                text_parts.append(str(item[1]))

        return correct_ocr_text(" ".join(text_parts))

    def _recognize_latex_with_pix2tex(self, image_file: Path) -> str:
        """Recognize a math expression from an image using pix2tex."""
        from PIL import Image

        image = Image.open(image_file).convert("RGB")
        latex = self.pix2tex_model(image)
        return str(latex).strip()

    def _recognize_with_hybrid(self, image_file: Path) -> dict:
        """
        Use EasyOCR for text rows and pix2tex for math-heavy rows.

        EasyOCR is used first as a cheap text signal. If the row looks like
        algebra, pix2tex is used and its LaTeX is normalized into plain text.
        """
        easy_text = self._recognize_with_easyocr(image_file)
        use_math_ocr = self._looks_like_math_row(easy_text, image_file)

        if use_math_ocr:
            try:
                latex = self._recognize_latex_with_pix2tex(image_file)
                plain_text = latex_to_plain_text(latex)

                if plain_text and not self._is_bad_pix2tex_output(latex, plain_text, easy_text):
                    return {
                        "text": plain_text,
                        "backend": "pix2tex",
                        "raw_latex": latex,
                        "easyocr_text": easy_text,
                    }

                return {
                    "text": easy_text,
                    "backend": "easyocr_fallback_rejected_pix2tex",
                    "raw_latex": "",
                    "easyocr_text": easy_text,
                    "ocr_warning": "pix2tex output rejected as likely hallucination",
                }
            except Exception as error:
                return {
                    "text": easy_text,
                    "backend": "easyocr_fallback_after_pix2tex_error",
                    "raw_latex": "",
                    "easyocr_text": easy_text,
                    "ocr_error": str(error),
                }

        return {
            "text": easy_text,
            "backend": "easyocr",
            "raw_latex": "",
            "easyocr_text": easy_text,
        }

    def _looks_like_math_row(self, easy_text: str, image_file: Path) -> bool:
        """
        Decide whether a cropped row should be handled by math OCR.

        The test combines OCR text hints with image-shape hints, because
        EasyOCR often misreads handwritten symbols before we get to pix2tex.
        """
        text = easy_text.strip()
        lower_text = text.lower()

        text_keywords = (
            "resolve",
            "solve",
            "let",
            "then",
            "therefore",
            "comparing",
            "coeff",
            "partial",
            "fraction",
            "fractions",
        )

        if any(keyword in lower_text for keyword in text_keywords):
            return False

        math_tokens = ("=", "+", "-", "*", "/", "^", "x", "a", "b", "frac", "sqrt")
        has_math_text = any(token in lower_text for token in math_tokens) and bool(re.search(r"\d|[xab]", lower_text))

        if has_math_text:
            return True

        try:
            import cv2
            import numpy as np

            image = cv2.imread(str(image_file), cv2.IMREAD_GRAYSCALE)

            if image is None:
                return False

            _, mask = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            height, width = mask.shape[:2]
            horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(20, width // 5), 1))
            horizontal_lines = cv2.morphologyEx(mask, cv2.MORPH_OPEN, horizontal_kernel)
            horizontal_pixels = int(np.sum(horizontal_lines > 0))
            ink_pixels = max(1, int(np.sum(mask > 0)))
            has_fraction_bar = horizontal_pixels / ink_pixels > 0.08
            is_wide_expression = width > height * 4 and ink_pixels > 80

            return has_fraction_bar or is_wide_expression
        except Exception:
            return False

    def _is_bad_pix2tex_output(self, latex: str, plain_text: str, easy_text: str) -> bool:
        """
        Reject pix2tex hallucinations.

        pix2tex is useful for clean isolated math expressions, but it can
        hallucinate huge scientific formulas when given mixed text, page lines,
        or noisy notebook crops. Those outputs are worse than EasyOCR, so the
        hybrid engine falls back to EasyOCR when this check fails.
        """
        latex_value = latex or ""
        plain_value = plain_text or ""
        easy_value = easy_text or ""

        suspicious_tokens = (
            "\\begin{array}",
            "\\mathrm",
            "\\langle",
            "\\nabla",
            "\\sigma",
            "\\omega",
            "\\varphi",
            "\\mathbf",
            "\\vec",
            "\\dagger",
            "\\star",
            "\\lambda",
            "\\partial",
            "\\varepsilon",
            "\\nonumber",
            "\\dots",
            "\\ldots",
            "\\qquad",
            "\\quad\\quad\\quad",
        )

        if any(token in latex_value for token in suspicious_tokens):
            return True

        command_count = latex_value.count("\\")
        quad_count = latex_value.count("\\quad")
        cdot_count = latex_value.count("\\cdot")

        if command_count > 18 or quad_count > 3 or cdot_count > 8:
            return True

        if len(plain_value) > 4 * max(1, len(easy_value)) and len(plain_value) > 80:
            return True

        if re.search(
            r"(beginarray|mathbf|mathrm|langle|ldots|qquad|mathbfnabla|mathbfsigma|quadquad)",
            plain_value,
        ):
            return True

        return False

    def _extract_text_parts(self, result) -> list[str]:
        """
        Extract text strings from common PaddleOCR result formats.

        PaddleOCR versions may return nested lists or dictionaries. This helper
        walks the result safely and collects recognized text fragments.
        """
        parts = []

        def walk(value) -> None:
            if value is None:
                return

            if isinstance(value, dict):
                for key in ("text", "rec_text", "transcription"):
                    if key in value and isinstance(value[key], str):
                        parts.append(value[key])
                for nested in value.values():
                    if isinstance(nested, (list, tuple, dict)):
                        walk(nested)
                return

            if (
                isinstance(value, (list, tuple))
                and len(value) >= 2
                and isinstance(value[0], str)
                and isinstance(value[1], (int, float))
            ):
                parts.append(value[0])
                return

            if isinstance(value, (list, tuple)):
                for item in value:
                    walk(item)

        walk(result)

        cleaned_parts = []
        for part in parts:
            cleaned = str(part).strip()
            if cleaned and cleaned not in cleaned_parts:
                cleaned_parts.append(cleaned)

        return cleaned_parts

    def _manual_recognize(self, image_file: Path) -> str:
        """
        Ask the user to type the text visible in a cropped step image.
        """
        print()
        print(f"Enter OCR text for {image_file.as_posix()}:")
        print("Tip: Open the image if needed, then type the algebra step exactly.")

        text = input("> ").strip()

        while text == "":
            print("OCR text cannot be empty. Please type the visible step.")
            text = input("> ").strip()

        return text


def latex_to_plain_text(latex: str) -> str:
    """
    Convert common pix2tex LaTeX output into simpler text for JSON.

    This is intentionally lightweight. The raw LaTeX is also preserved in the
    step metadata so Module 02 can use either representation later.
    """
    if not latex:
        return ""

    text = str(latex).strip()
    text = text.replace("\\left", "").replace("\\right", "")
    text = text.replace("\\,", " ").replace("\\;", " ").replace("\\!", "")
    text = text.replace("\\cdot", "*").replace("\\times", "*").replace("\\div", "/")
    text = text.replace("\\pm", "+/-")
    text = _replace_latex_fractions(text)
    text = re.sub(r"\^\{([^{}]+)\}", r"^\1", text)
    text = re.sub(r"_\{([^{}]+)\}", r"_\1", text)
    text = text.replace("{", "").replace("}", "")
    text = text.replace("\\", "")
    text = correct_ocr_text(text)
    return text


def _replace_latex_fractions(text: str) -> str:
    """Replace simple LaTeX fractions with (numerator)/(denominator)."""
    while "\\frac" in text:
        start = text.find("\\frac")
        numerator_start = start + len("\\frac")

        numerator, numerator_end = _read_latex_group(text, numerator_start)
        denominator, denominator_end = _read_latex_group(text, numerator_end)

        if numerator is None or denominator is None:
            break

        replacement = f"({numerator})/({denominator})"
        text = text[:start] + replacement + text[denominator_end:]

    return text


def _read_latex_group(text: str, start: int) -> tuple[str | None, int]:
    """Read a {...} group from a LaTeX string."""
    while start < len(text) and text[start].isspace():
        start += 1

    if start >= len(text) or text[start] != "{":
        return None, start

    depth = 0
    content_start = start + 1

    for index in range(start, len(text)):
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1

            if depth == 0:
                return text[content_start:index], index + 1

    return None, start


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run OCR on cropped step images.")
    parser.add_argument(
        "images",
        nargs="+",
        help="One or more cropped step image paths.",
    )
    parser.add_argument(
        "--auto",
        action="store_true",
        help="Use automatic OCR instead of manual typing.",
    )
    parser.add_argument(
        "--backend",
        choices=["auto", "paddle", "easyocr", "pix2tex", "hybrid", "manual"],
        default="manual",
        help="OCR backend to use.",
    )
    args = parser.parse_args()

    selected_backend = "auto" if args.auto else args.backend
    ocr_engine = OCREngine(backend=selected_backend, manual_fallback=selected_backend == "manual")

    for image in args.images:
        result = ocr_engine.recognize_step_details(image)
        print(result)
