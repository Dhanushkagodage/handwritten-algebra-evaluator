"""
Visualizer module for generating publication-quality (300 DPI) plot figures.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional
import matplotlib
matplotlib.use("Agg")  # Non-interactive background rendering
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from config import FIGURES_DIR, PLOT_DPI

logger = logging.getLogger(__name__)

# Aesthetic configuration
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "axes.labelweight": "bold",
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 14,
    "figure.autolayout": True,
})


class Visualizer:
    """Generates focused statistical plot figures saved as high-res 300 DPI PNG files."""

    def __init__(self, output_dir: Path = FIGURES_DIR, dpi: int = PLOT_DPI):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.dpi = dpi

    def generate_all_figures(
        self,
        df_question: pd.DataFrame,
        df_step: pd.DataFrame,
        validity_cm: np.ndarray,
        validity_labels: List[str],
        scheme_cm: np.ndarray,
        scheme_labels: List[str],
    ) -> List[Path]:
        """Generates required 6 publication-grade figures."""
        generated_paths = []

        # 1. Histogram of Absolute Errors
        p1 = self.plot_error_histogram(df_question)
        generated_paths.append(p1)

        # 2. Bland-Altman Plot
        p2 = self.plot_bland_altman(df_question)
        generated_paths.append(p2)

        # 3. Box Plot of Mark Differences
        p3 = self.plot_box_plot_errors(df_question, df_step)
        generated_paths.append(p3)

        # 4. Validity Confusion Matrix Heatmap
        p4 = self.plot_confusion_heatmap(validity_cm, validity_labels, "Step Validity Confusion Matrix", "validity_confusion.png")
        generated_paths.append(p4)

        # 5. Scheme Matching Confusion Matrix Heatmap
        p5 = self.plot_confusion_heatmap(scheme_cm, scheme_labels, "Scheme Matching Confusion Matrix", "scheme_confusion.png")
        generated_paths.append(p5)

        # 6. Correlation Heatmap
        p6 = self.plot_correlation_heatmap(df_question)
        generated_paths.append(p6)

        logger.info(f"Successfully generated {len(generated_paths)} figures in {self.output_dir}")
        return generated_paths

    def plot_error_histogram(self, df_q: pd.DataFrame) -> Path:
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.histplot(df_q["absolute_error"], kde=True, bins=12, color="#3498db", edgecolor="black", ax=ax)
        mean_err = df_q["absolute_error"].mean()
        ax.axvline(mean_err, color="#e74c3c", linestyle="--", linewidth=2, label=f"MAE = {mean_err:.2f}")
        ax.set_title("Histogram of Absolute Errors")
        ax.set_xlabel("Absolute Error |GT - System|")
        ax.set_ylabel("Question Count")
        ax.legend()
        path = self.output_dir / "error_histogram.png"
        fig.savefig(path, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        return path

    def plot_bland_altman(self, df_q: pd.DataFrame) -> Path:
        fig, ax = plt.subplots(figsize=(8, 6))
        gt = df_q["gt_total_marks"]
        sys = df_q["sys_total_marks"]
        means = (gt + sys) / 2.0
        diffs = sys - gt
        mean_diff = np.mean(diffs)
        std_diff = np.std(diffs)

        ax.scatter(means, diffs, color="#2980b9", alpha=0.8, edgecolors="k", s=70)
        ax.axhline(mean_diff, color="#e74c3c", linestyle="-", linewidth=2, label=f"Mean Bias = {mean_diff:.2f}")
        ax.axhline(mean_diff + 1.96 * std_diff, color="#27ae60", linestyle="--", linewidth=1.5, label=f"+1.96 SD = {mean_diff + 1.96 * std_diff:.2f}")
        ax.axhline(mean_diff - 1.96 * std_diff, color="#27ae60", linestyle="--", linewidth=1.5, label=f"-1.96 SD = {mean_diff - 1.96 * std_diff:.2f}")

        ax.set_title("Bland–Altman Plot: System Agreement with Teacher")
        ax.set_xlabel("Mean of Ground Truth and AI System Marks")
        ax.set_ylabel("Difference (AI System - Teacher)")
        ax.legend(loc="upper right")
        path = self.output_dir / "bland_altman.png"
        fig.savefig(path, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        return path

    def plot_box_plot_errors(self, df_q: pd.DataFrame, df_s: pd.DataFrame) -> Path:
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        
        df_q_copy = df_q.copy()
        df_q_copy["mark_diff"] = df_q_copy["sys_total_marks"] - df_q_copy["gt_total_marks"]
        sns.boxplot(y=df_q_copy["mark_diff"], color="#9b59b6", ax=axes[0])
        axes[0].set_title("Question Mark Difference (Sys - GT)")
        axes[0].set_ylabel("Mark Difference")
        axes[0].axhline(0, color="gray", linestyle="--")

        sns.boxplot(y=df_s["mark_diff"], color="#1abc9c", ax=axes[1])
        axes[1].set_title("Step Mark Difference (GT - Sys)")
        axes[1].set_ylabel("Mark Difference")
        axes[1].axhline(0, color="gray", linestyle="--")

        path = self.output_dir / "box_plot_errors.png"
        fig.savefig(path, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        return path

    def plot_confusion_heatmap(self, cm: np.ndarray, labels: List[str], title: str, filename: str) -> Path:
        fig, ax = plt.subplots(figsize=(7, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels, ax=ax, cbar=False, linewidths=1)
        ax.set_title(title)
        ax.set_xlabel("Predicted (AI System)")
        ax.set_ylabel("Ground Truth (Teacher)")
        path = self.output_dir / filename
        fig.savefig(path, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        return path

    def plot_correlation_heatmap(self, df_q: pd.DataFrame) -> Path:
        fig, ax = plt.subplots(figsize=(7, 6))
        corr_cols = ["gt_total_marks", "sys_total_marks", "max_marks", "absolute_error", "squared_error"]
        corr_matrix = df_q[corr_cols].corr()
        
        sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", ax=ax, linewidths=1)
        ax.set_title("Correlation Heatmap")
        path = self.output_dir / "correlation_heatmap.png"
        fig.savefig(path, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        return path
