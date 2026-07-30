"""
Configuration settings for the AI-Based Stepwise Algebra Evaluation System Analysis Suite.

Defines input/output file paths, plot aesthetics, topic classification patterns,
and statistical significance parameters.
"""

from pathlib import Path
from typing import Dict, List, Pattern
import re

# Base Directories
BASE_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_DIR = BASE_DIR / "3_output_results"
DEFAULT_OUTPUT_DIR = BASE_DIR / "results"

# Input CSV Paths
QUESTION_LEVEL_CSV = DEFAULT_INPUT_DIR / "question_level_results.csv"
STEP_LEVEL_CSV = DEFAULT_INPUT_DIR / "step_level_results.csv"

# Output Directories
FIGURES_DIR = DEFAULT_OUTPUT_DIR / "figures"
METRICS_DIR = DEFAULT_OUTPUT_DIR / "metrics"
REPORT_MD_PATH = DEFAULT_OUTPUT_DIR / "report.md"
REPORT_PDF_PATH = DEFAULT_OUTPUT_DIR / "report.pdf"

# Statistical Settings
SIGNIFICANCE_ALPHA: float = 0.05

# Visualization Settings
PLOT_DPI: int = 300
PLOT_STYLE: str = "seaborn-v0_8-whitegrid"
COLOR_PALETTE: str = "viridis"

# Topic Regex Rules for Algebra Topic Extraction
TOPIC_PATTERNS: Dict[str, List[str]] = {
    "Quadratic Equations": [
        r"completing the square", r"quadratic", r"x²", r"z²", r"roots of"
    ],
    "Simultaneous Equations": [
        r"simultaneously", r"system of equations", r"y = 2x"
    ],
    "Partial Fractions": [
        r"partial fractions", r"decompose"
    ],
    "Logarithms": [
        r"logarithm", r"log₃", r"log₂", r"natural logarithm", r"ln\("
    ],
    "Binomial Expansion": [
        r"binomial", r"ascending powers", r"expansion"
    ],
    "Matrices": [
        r"matrix", r"matrices", r"determinant", r"eigenvalue", r"eigenvector", r"AX = B"
    ],
    "Polynomial Division": [
        r"divide f\(x\)", r"factor theorem", r"long division", r"remainder"
    ],
    "Mathematical Induction": [
        r"induction", r"prove by mathematical induction", r"positive integers n"
    ],
    "Complex Numbers": [
        r"De Moivre", r"complex quadratic", r"locus", r"i√3", r"arg\(z\)"
    ],
    "Trigonometry": [
        r"R cos", r"sin\(x\)", r"cos\(2x\)", r"R sin"
    ],
    "Vectors & Planes": [
        r"normal vector", r"line r =", r"plane", r"cross product"
    ],
    "Series & Sequences": [
        r"geometric series", r"Maclaurin series", r"sum to infinity"
    ],
    "Hyperbolic Functions": [
        r"cosh", r"sinh", r"tanh"
    ]
}
