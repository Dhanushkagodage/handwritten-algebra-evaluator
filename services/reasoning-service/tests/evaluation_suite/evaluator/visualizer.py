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
        """Generates publication-grade figures showing full data distributions."""
        generated_paths = []

        # 0. System-Assigned Marks vs Human-Assigned Marks (Full Data Visualization)
        p0 = self.plot_system_vs_human_marks(df_question)
        generated_paths.extend(p0)

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

    def plot_system_vs_human_marks(self, df_q: pd.DataFrame) -> List[Path]:
        """
        Generates plots comparing System-Assigned Marks against Human-Assigned Marks (Teacher GT).
        Ensures ALL data points are explicitly visualized without hiding overlapping points.
        """
        paths = []
        
        gt = df_q["gt_total_marks"].values
        sys_marks = df_q["sys_total_marks"].values
        tc_ids = df_q["test_case_id"].values if "test_case_id" in df_q.columns else [f"Q{i+1}" for i in range(len(df_q))]
        
        # Add deterministic jitter so overlapping scores (e.g., 8.0 vs 8.0) are visibly separated
        np.random.seed(42)  # Consistent layout
        jitter_x = gt + np.random.uniform(-0.08, 0.08, size=len(gt))
        jitter_y = sys_marks + np.random.uniform(-0.08, 0.08, size=len(sys_marks))
        
        # Color points by error magnitude
        abs_err = np.abs(sys_marks - gt)
        colors_list = []
        for err in abs_err:
            if err == 0:
                colors_list.append("#2ecc71")  # Green for exact match
            elif err <= 1.0:
                colors_list.append("#3498db")  # Blue for small error
            else:
                colors_list.append("#e74c3c")  # Red for larger error

        max_val = max(np.max(gt), np.max(sys_marks)) + 1.0

        from matplotlib.patches import Patch

        legend_elements = [
            Patch(facecolor="#2ecc71", edgecolor="black", label=f"Exact Match (n={(abs_err == 0).sum()})"),
            Patch(facecolor="#3498db", edgecolor="black", label=f"Small Error ≤ 1.0 (n={((abs_err > 0) & (abs_err <= 1.0)).sum()})"),
            Patch(facecolor="#e74c3c", edgecolor="black", label=f"Error > 1.0 (n={(abs_err > 1.0).sum()})"),
            plt.Line2D([0], [0], color="#7f8c8d", linestyle="--", linewidth=2, label="Perfect Agreement (y = x)")
        ]
        
        # --- Figure 1: Scatter Plot with Jitter & Annotations for All Data Points ---
        fig, ax = plt.subplots(figsize=(9, 7))
        scatter = ax.scatter(
            jitter_x,
            jitter_y,
            c=colors_list,
            s=90,
            alpha=0.85,
            edgecolors="black",
            linewidth=1.0,
            zorder=3
        )
        
        # Reference Line (y = x)
        ax.plot([0, max_val], [0, max_val], color="#7f8c8d", linestyle="--", linewidth=2, label="Ideal Line (y = x)", zorder=2)
        
        # Annotate each data point with test case ID so all N data points are explicitly identified
        for i, tc_id in enumerate(tc_ids):
            ax.annotate(
                tc_id,
                (jitter_x[i], jitter_y[i]),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=8,
                alpha=0.8,
                fontweight="bold"
            )
            
        ax.legend(handles=legend_elements, loc="upper left", frameon=True)
        
        r = df_q["gt_total_marks"].corr(df_q["sys_total_marks"])
        mae = abs_err.mean()
        ax.text(
            0.95, 0.05,
            f"Total Questions (N): {len(df_q)}\nPearson r: {r:.4f}\nMAE: {mae:.2f} marks",
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="bottom",
            horizontalalignment="right",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.9, edgecolor="#bdc3c7")
        )
        
        ax.set_title(f"System-Assigned Marks vs. Human-Assigned Marks (All N={len(df_q)} Questions)", fontsize=13, pad=12)
        ax.set_xlabel("Human-Assigned Marks (Teacher Ground Truth)", fontweight="bold")
        ax.set_ylabel("System-Assigned Marks (AI System)", fontweight="bold")
        ax.set_xlim(-0.5, max_val)
        ax.set_ylim(-0.5, max_val)
        
        p_scatter = self.output_dir / "system_vs_human_scatter.png"
        p_scatter_alias = self.output_dir / "scatter_marks.png"
        fig.savefig(p_scatter, dpi=self.dpi, bbox_inches="tight")
        fig.savefig(p_scatter_alias, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        paths.extend([p_scatter, p_scatter_alias])

        # --- Figure 2: Grouped Bar Chart Displaying ALL Questions Side-by-Side ---
        fig, ax = plt.subplots(figsize=(14, 6))
        x = np.arange(len(df_q))
        width = 0.38

        rects1 = ax.bar(x - width/2, df_q["gt_total_marks"], width, label="Human-Assigned (Teacher GT)", color="#2980b9", edgecolor="black", alpha=0.85)
        rects2 = ax.bar(x + width/2, df_q["sys_total_marks"], width, label="System-Assigned (AI)", color="#2ecc71", edgecolor="black", alpha=0.85)

        ax.set_title(f"Human-Assigned vs. System-Assigned Marks by Question (All {len(df_q)} Test Cases)", fontsize=13, pad=12)
        ax.set_xlabel("Question / Test Case ID", fontweight="bold")
        ax.set_ylabel("Total Marks", fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(tc_ids, rotation=45, ha="right", fontsize=9)
        ax.legend(loc="upper right", frameon=True)
        ax.set_ylim(0, max_val + 1.5)

        # Value annotations on bars
        for rect in rects1:
            h = rect.get_height()
            ax.annotate(f"{h:.1f}", xy=(rect.get_x() + rect.get_width()/2, h), xytext=(0, 2),
                        textcoords="offset points", ha="center", va="bottom", fontsize=7.5, rotation=90)
        for rect in rects2:
            h = rect.get_height()
            ax.annotate(f"{h:.1f}", xy=(rect.get_x() + rect.get_width()/2, h), xytext=(0, 2),
                        textcoords="offset points", ha="center", va="bottom", fontsize=7.5, rotation=90)

        p_bar = self.output_dir / "system_vs_human_by_question.png"
        fig.savefig(p_bar, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        paths.append(p_bar)

        # --- Figure 3: Combined 2-Panel Master Figure ---
        fig = plt.figure(figsize=(16, 7))
        gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.3])
        
        # Subplot A: Scatter
        ax1 = fig.add_subplot(gs[0])
        ax1.scatter(jitter_x, jitter_y, c=colors_list, s=80, alpha=0.85, edgecolors="black", zorder=3)
        ax1.plot([0, max_val], [0, max_val], color="#7f8c8d", linestyle="--", linewidth=2, zorder=2)
        for i, tc_id in enumerate(tc_ids):
            ax1.annotate(tc_id, (jitter_x[i], jitter_y[i]), xytext=(3, 3), textcoords="offset points", fontsize=7.5, alpha=0.8)
        ax1.set_title("A. Scatter Plot (All Data Points with Jitter)", fontsize=11, fontweight="bold")
        ax1.set_xlabel("Human Ground Truth Marks", fontweight="bold")
        ax1.set_ylabel("System-Assigned Marks", fontweight="bold")
        ax1.legend(handles=legend_elements, loc="upper left", fontsize=8)
        ax1.set_xlim(-0.5, max_val)
        ax1.set_ylim(-0.5, max_val)
        
        # Subplot B: Bar
        ax2 = fig.add_subplot(gs[1])
        ax2.bar(x - width/2, df_q["gt_total_marks"], width, label="Human (GT)", color="#2980b9", edgecolor="black", alpha=0.85)
        ax2.bar(x + width/2, df_q["sys_total_marks"], width, label="System (AI)", color="#2ecc71", edgecolor="black", alpha=0.85)
        ax2.set_title(f"B. Question-by-Question Comparison (All N={len(df_q)})", fontsize=11, fontweight="bold")
        ax2.set_xlabel("Test Case ID", fontweight="bold")
        ax2.set_ylabel("Total Marks", fontweight="bold")
        ax2.set_xticks(x)
        ax2.set_xticklabels(tc_ids, rotation=45, ha="right", fontsize=8)
        ax2.legend(loc="upper right", fontsize=8)
        ax2.set_ylim(0, max_val + 1.5)
        
        fig.suptitle("System-Assigned Marks against Human-Assigned Marks (Complete Dataset Visualization)", fontsize=14, fontweight="bold", y=0.98)
        
        p_master = self.output_dir / "system_vs_human_marks.png"
        fig.savefig(p_master, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        paths.append(p_master)

        return paths

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

