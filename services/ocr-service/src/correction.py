import argparse
import re


MATH_SYMBOLS = ("=", "+", "-", "*", "/", "^", "(", ")")
MATH_WORDS = ("log", "sqrt", "sin", "cos", "tan")


def _replace_common_math_symbols(text: str) -> str:
    """Replace common OCR/unicode math symbols with simple keyboard symbols."""
    replacements = {
        "×": "*",
        "÷": "/",
        "−": "-",
        "–": "-",
        "—": "-",
        "＝": "=",
        "＋": "+",
        "²": "^2",
        "³": "^3",
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    return text


def _normalize_variable_x(text: str) -> str:
    """
    Convert capital X to lowercase x.

    This is useful because OCR often reads handwritten x as X.
    """
    return text.replace("X", "x")


def _replace_o_with_zero_in_math_context(text: str) -> str:
    """
    Replace uppercase O with 0 only when it looks like part of an equation.

    This avoids changing natural language words such as "OR" or "ONLY".
    """
    has_math_context = any(symbol in text for symbol in MATH_SYMBOLS) or bool(re.search(r"\d", text))

    if not has_math_context:
        return text

    text = re.sub(r"(?<=[0-9=+\-*/^().])O", "0", text)
    text = re.sub(r"O(?=[0-9=+\-*/^().])", "0", text)
    text = re.sub(r"(?<![A-Za-z])O(?![A-Za-z])", "0", text)

    return text


def _normalize_operator_spacing(text: str) -> str:
    """Add consistent spacing around common algebra operators."""
    text = re.sub(r"\s*=\s*", " = ", text)
    text = re.sub(r"\s*\+\s*", " + ", text)
    text = re.sub(r"\s*-\s*", " - ", text)
    text = re.sub(r"\s*\*\s*", " * ", text)
    text = re.sub(r"\s*/\s*", " / ", text)
    text = re.sub(r"\s*\^\s*", "^", text)

    text = re.sub(r"\(\s+", "(", text)
    text = re.sub(r"\s+\)", ")", text)

    return text


def correct_ocr_text(text: str) -> str:
    """
    Apply safe algebra-specific OCR correction rules.

    The goal is not to solve or mark the answer. This function only cleans the
    recognized text so the next module receives a more consistent expression.
    """
    if text is None:
        return ""

    corrected = str(text).strip()
    corrected = _replace_common_math_symbols(corrected)
    corrected = _normalize_variable_x(corrected)
    corrected = _replace_o_with_zero_in_math_context(corrected)
    corrected = _normalize_operator_spacing(corrected)
    corrected = re.sub(r"\s+", " ", corrected)

    return corrected.strip()


def detect_step_type(text: str) -> str:
    """
    Detect whether a recognized step is mathematical or plain text.

    Returns:
        "math" when algebra symbols, variables, numbers, or math words appear.
        "text" otherwise.
    """
    if text is None:
        return "text"

    value = str(text).strip()

    if value == "":
        return "text"

    lower_value = value.lower()

    if any(symbol in value for symbol in ("=", "+", "-", "*", "/", "^")):
        return "math"

    if re.search(r"\b[xyn]\b", lower_value):
        return "math"

    if any(word in lower_value for word in MATH_WORDS):
        return "math"

    if re.search(r"\d", value):
        return "math"

    return "text"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Correct OCR text for algebra answer steps.")
    parser.add_argument(
        "text",
        nargs="+",
        help="OCR text to correct.",
    )
    args = parser.parse_args()

    raw_text = " ".join(args.text)
    corrected_text = correct_ocr_text(raw_text)

    print(f"Raw text: {raw_text}")
    print(f"Corrected text: {corrected_text}")
    print(f"Step type: {detect_step_type(corrected_text)}")
