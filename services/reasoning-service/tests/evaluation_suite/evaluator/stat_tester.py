"""
Statistical testing module for hypothesis testing (Paired t-test & Wilcoxon signed-rank test).
"""

import logging
from typing import Any, Dict
import numpy as np
import pandas as pd
from scipy import stats

from config import SIGNIFICANCE_ALPHA

logger = logging.getLogger(__name__)


class StatTester:
    """Performs inferential statistical hypothesis testing between Teacher GT and AI System."""

    def __init__(self, alpha: float = SIGNIFICANCE_ALPHA):
        self.alpha = alpha

    def test_marks_difference(self, gt_marks: np.ndarray, sys_marks: np.ndarray) -> Dict[str, Any]:
        """Runs Paired t-test and Wilcoxon signed-rank test on question totals."""
        n_samples = len(gt_marks)
        diffs = sys_marks - gt_marks
        mean_diff = float(np.mean(diffs))
        std_diff = float(np.std(diffs, ddof=1)) if n_samples > 1 else 0.0

        # Paired t-test
        t_stat, t_p_val = stats.ttest_rel(gt_marks, sys_marks)

        # Wilcoxon Signed-Rank Test
        if np.all(diffs == 0):
            w_stat, w_p_val = 0.0, 1.0
        else:
            w_stat, w_p_val = stats.wilcoxon(gt_marks, sys_marks)

        # Effect Size (Cohen's d for paired samples)
        cohen_d = float(mean_diff / std_diff) if std_diff > 0 else 0.0

        # Decision
        is_significant_ttest = bool(t_p_val < self.alpha)
        is_significant_wilcoxon = bool(w_p_val < self.alpha)

        if is_significant_ttest or is_significant_wilcoxon:
            conclusion = (
                f"Statistically significant difference detected between AI System and Teacher marks "
                f"(Paired t-test p={t_p_val:.4f}, Wilcoxon p={w_p_val:.4f} at alpha={self.alpha})."
            )
        else:
            conclusion = (
                f"No statistically significant difference detected between AI System and Teacher marks "
                f"(Paired t-test p={t_p_val:.4f}, Wilcoxon p={w_p_val:.4f} at alpha={self.alpha}). "
                f"The AI grading aligns with human teacher grading."
            )

        results = {
            "n_samples": n_samples,
            "mean_difference": mean_diff,
            "std_difference": std_diff,
            "paired_t_statistic": float(t_stat),
            "paired_t_p_value": float(t_p_val),
            "is_significant_ttest": is_significant_ttest,
            "wilcoxon_statistic": float(w_stat),
            "wilcoxon_p_value": float(w_p_val),
            "is_significant_wilcoxon": is_significant_wilcoxon,
            "cohens_d": cohen_d,
            "alpha_threshold": self.alpha,
            "conclusion": conclusion,
        }
        return results
