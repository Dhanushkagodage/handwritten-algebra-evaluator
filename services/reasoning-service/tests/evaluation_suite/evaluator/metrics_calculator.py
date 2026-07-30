"""
Metrics calculator module for quantitative evaluation of question-level,
step-level validity, scheme matching, and step marking.
"""

import logging
from typing import Any, Dict, Optional
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)

logger = logging.getLogger(__name__)


class MetricsCalculator:
    """Calculates concise, focused quantitative metrics for AI grading vs teacher ground truth."""

    def __init__(self, df_question: pd.DataFrame, df_step: pd.DataFrame):
        self.df_question = df_question.copy()
        self.df_step = df_step.copy()

    def compute_question_metrics(self) -> Dict[str, Any]:
        """Computes focused question-level evaluation metrics."""
        gt = self.df_question["gt_total_marks"].values
        sys = self.df_question["sys_total_marks"].values

        mae = float(mean_absolute_error(gt, sys))
        mse = float(mean_squared_error(gt, sys))
        rmse = float(np.sqrt(mse))

        if len(gt) > 1 and np.std(gt) > 0 and np.std(sys) > 0:
            pearson_r, _ = stats.pearsonr(gt, sys)
        else:
            pearson_r = 0.0

        exact_match_acc = float(np.mean(gt == sys) * 100.0)
        mark_diffs = sys - gt
        mbe = float(np.mean(mark_diffs))
        r2 = float(r2_score(gt, sys))
        exact_match_f1 = float(f1_score(gt == sys, np.ones_like(gt, dtype=bool), average="binary", zero_division=0))

        return {
            "MAE": mae,
            "Pearson Correlation (r)": float(pearson_r),
            "Exact Match Accuracy (%)": exact_match_acc,
            "Exact Match F1": exact_match_f1,
            "Mean Bias Error (MBE)": mbe,
            "RMSE": rmse,
            "R² Score": r2,
        }

    def compute_step_validity_metrics(self) -> Dict[str, Any]:
        """Evaluates step-level validity classification."""
        gt_validity = self.df_step["gt_validity"].astype(str)
        sys_validity = self.df_step["sys_validity"].astype(str)
        labels = sorted(list(set(gt_validity).union(set(sys_validity))))

        kappa = float(cohen_kappa_score(gt_validity, sys_validity))
        f1_macro = float(f1_score(gt_validity, sys_validity, average="macro", zero_division=0))
        f1_w = float(f1_score(gt_validity, sys_validity, average="weighted", zero_division=0))
        f1_micro = float(f1_score(gt_validity, sys_validity, average="micro", zero_division=0))
        acc = float(accuracy_score(gt_validity, sys_validity))
        prec = float(precision_score(gt_validity, sys_validity, average="weighted", zero_division=0))
        rec = float(recall_score(gt_validity, sys_validity, average="weighted", zero_division=0))

        # Per-class F1 scores
        per_class_f1 = f1_score(gt_validity, sys_validity, labels=labels, average=None, zero_division=0)
        per_class_f1_dict = {f"F1 ({lbl})": float(score) for lbl, score in zip(labels, per_class_f1)}

        cm = confusion_matrix(gt_validity, sys_validity, labels=labels)

        res = {
            "Macro F1": f1_macro,
            "Weighted F1": f1_w,
            "Micro F1": f1_micro,
        }
        res.update(per_class_f1_dict)
        res.update({
            "Cohen's κ": kappa,
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "labels": labels,
            "confusion_matrix": cm,
        })
        return res

    def compute_scheme_matching_metrics(self) -> Dict[str, Any]:
        """Evaluates scheme matching alignment."""
        gt_scheme = self.df_step["gt_matched_scheme"].astype(str)
        sys_scheme = self.df_step["sys_matched_scheme"].astype(str)
        labels = sorted(list(set(gt_scheme).union(set(sys_scheme))))

        cm = confusion_matrix(gt_scheme, sys_scheme, labels=labels)
        acc = float(accuracy_score(gt_scheme, sys_scheme))
        f1_macro = float(f1_score(gt_scheme, sys_scheme, average="macro", zero_division=0))
        f1_w = float(f1_score(gt_scheme, sys_scheme, average="weighted", zero_division=0))
        f1_micro = float(f1_score(gt_scheme, sys_scheme, average="micro", zero_division=0))
        kappa = float(cohen_kappa_score(gt_scheme, sys_scheme))

        return {
            "Macro F1": f1_macro,
            "Weighted F1": f1_w,
            "Micro F1": f1_micro,
            "Accuracy": acc,
            "Cohen's κ": kappa,
            "labels": labels,
            "confusion_matrix": cm,
        }

    def compute_step_marking_metrics(self) -> Dict[str, Any]:
        """Evaluates step-level mark allocation accuracy."""
        gt_marks = self.df_step["gt_marks"].values
        sys_marks = self.df_step["sys_marks"].values

        mae = float(mean_absolute_error(gt_marks, sys_marks))
        rmse = float(np.sqrt(mean_squared_error(gt_marks, sys_marks)))

        if len(gt_marks) > 1 and np.std(gt_marks) > 0 and np.std(sys_marks) > 0:
            pearson_r, _ = stats.pearsonr(gt_marks, sys_marks)
        else:
            pearson_r = 0.0

        mean_diff = float(np.mean(sys_marks - gt_marks))

        return {
            "Step MAE": mae,
            "Step RMSE": rmse,
            "Pearson r": float(pearson_r),
            "Mean Difference": mean_diff,
        }
