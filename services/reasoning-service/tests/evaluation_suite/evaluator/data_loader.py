"""
Data loader module for reading, cleaning, and preprocessing question-level
and step-level evaluation results CSV files.
"""

import logging
import re
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd

from config import QUESTION_LEVEL_CSV, STEP_LEVEL_CSV, TOPIC_PATTERNS

logger = logging.getLogger(__name__)


class DataLoader:
    """Handles loading, validation, and pre-processing of evaluation datasets."""

    def __init__(
        self,
        question_csv: Path = QUESTION_LEVEL_CSV,
        step_csv: Path = STEP_LEVEL_CSV,
    ):
        self.question_csv = Path(question_csv)
        self.step_csv = Path(step_csv)
        self.df_question: pd.DataFrame = pd.DataFrame()
        self.df_step: pd.DataFrame = pd.DataFrame()

    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Loads both CSV files, handles missing values, and adds derived columns."""
        logger.info(f"Loading question-level data from {self.question_csv}")
        if not self.question_csv.exists():
            raise FileNotFoundError(f"Question-level CSV not found at: {self.question_csv}")

        self.df_question = pd.read_csv(self.question_csv)
        self._preprocess_question_data()

        logger.info(f"Loading step-level data from {self.step_csv}")
        if not self.step_csv.exists():
            raise FileNotFoundError(f"Step-level CSV not found at: {self.step_csv}")

        self.df_step = pd.read_csv(self.step_csv)
        self._preprocess_step_data()

        return self.df_question, self.df_step

    def _preprocess_question_data(self) -> None:
        """Preprocesses question-level dataset."""
        required_cols = [
            "test_case_id", "question_text", "max_marks",
            "gt_total_marks", "sys_total_marks"
        ]
        for col in required_cols:
            if col not in self.df_question.columns:
                raise ValueError(f"Missing required column '{col}' in question-level dataset")

        # Numeric conversions
        self.df_question["max_marks"] = pd.to_numeric(self.df_question["max_marks"], errors="coerce").fillna(0.0)
        self.df_question["gt_total_marks"] = pd.to_numeric(self.df_question["gt_total_marks"], errors="coerce").fillna(0.0)
        self.df_question["sys_total_marks"] = pd.to_numeric(self.df_question["sys_total_marks"], errors="coerce").fillna(0.0)

        # Derived metrics if missing
        if "absolute_error" not in self.df_question.columns:
            self.df_question["absolute_error"] = (self.df_question["gt_total_marks"] - self.df_question["sys_total_marks"]).abs()

        if "squared_error" not in self.df_question.columns:
            self.df_question["squared_error"] = (self.df_question["gt_total_marks"] - self.df_question["sys_total_marks"]) ** 2

        if "exact_match" not in self.df_question.columns:
            self.df_question["exact_match"] = self.df_question["gt_total_marks"] == self.df_question["sys_total_marks"]
        else:
            self.df_question["exact_match"] = self.df_question["exact_match"].astype(bool)

        # Topic Extraction
        self.df_question["extracted_topic"] = self.df_question["question_text"].apply(self._classify_topic)

    def _preprocess_step_data(self) -> None:
        """Preprocesses step-level dataset."""
        required_cols = [
            "test_case_id", "step_id", "gt_validity", "sys_validity", "gt_marks", "sys_marks"
        ]
        for col in required_cols:
            if col not in self.df_step.columns:
                raise ValueError(f"Missing required column '{col}' in step-level dataset")

        # Clean scheme match identifiers
        if "gt_matched_scheme" in self.df_step.columns:
            self.df_step["gt_matched_scheme"] = self.df_step["gt_matched_scheme"].fillna("unmatched").astype(str).str.replace(".0", "", regex=False)
        else:
            self.df_step["gt_matched_scheme"] = "unmatched"

        if "sys_matched_scheme" in self.df_step.columns:
            self.df_step["sys_matched_scheme"] = self.df_step["sys_matched_scheme"].fillna("unmatched").astype(str).str.replace(".0", "", regex=False)
        else:
            self.df_step["sys_matched_scheme"] = "unmatched"

        # Mark calculations
        self.df_step["gt_marks"] = pd.to_numeric(self.df_step["gt_marks"], errors="coerce").fillna(0.0)
        self.df_step["sys_marks"] = pd.to_numeric(self.df_step["sys_marks"], errors="coerce").fillna(0.0)
        self.df_step["mark_diff"] = self.df_step["gt_marks"] - self.df_step["sys_marks"]

        # Clean validity strings
        self.df_step["gt_validity"] = self.df_step["gt_validity"].astype(str).str.strip().str.lower()
        self.df_step["sys_validity"] = self.df_step["sys_validity"].astype(str).str.strip().str.lower()

    @staticmethod
    def _classify_topic(question_text: str) -> str:
        """Classifies question text into mathematical algebra topic using regex rules."""
        if not isinstance(question_text, str):
            return "General Algebra"

        for topic, patterns in TOPIC_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, question_text, re.IGNORECASE):
                    return topic

        return "General Algebra"
