"""
Report generator module for exporting metric CSVs and building Markdown (report.md)
and PDF (report.pdf) thesis evaluation reports.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

from config import METRICS_DIR, REPORT_MD_PATH, REPORT_PDF_PATH

logger = logging.getLogger(__name__)


class ReportGenerator:
    """Generates clean CSV metrics, Markdown thesis report, and PDF report."""

    def __init__(
        self,
        metrics_dir: Path = METRICS_DIR,
        report_md_path: Path = REPORT_MD_PATH,
        report_pdf_path: Path = REPORT_PDF_PATH,
    ):
        self.metrics_dir = Path(metrics_dir)
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        self.report_md_path = Path(report_md_path)
        self.report_pdf_path = Path(report_pdf_path)

    def export_csv_metrics(
        self,
        q_metrics: Dict[str, Any],
        step_val_metrics: Dict[str, Any],
        scheme_metrics: Dict[str, Any],
        step_mark_metrics: Dict[str, Any],
    ) -> None:
        """Exports metrics to individual CSV files."""
        # 1. Question Metrics CSV
        pd.DataFrame([q_metrics]).T.reset_index().rename(
            columns={"index": "metric", 0: "value"}
        ).to_csv(self.metrics_dir / "question_metrics.csv", index=False)

        # 2. Step Validity Metrics CSV
        val_summary = {k: v for k, v in step_val_metrics.items() if not isinstance(v, (pd.DataFrame, list, dict, np.ndarray))}
        pd.DataFrame([val_summary]).T.reset_index().rename(
            columns={"index": "metric", 0: "value"}
        ).to_csv(self.metrics_dir / "step_metrics.csv", index=False)

        # 3. Scheme Matching Metrics CSV
        scheme_summary = {k: v for k, v in scheme_metrics.items() if not isinstance(v, (pd.DataFrame, list, dict, np.ndarray))}
        pd.DataFrame([scheme_summary]).T.reset_index().rename(
            columns={"index": "metric", 0: "value"}
        ).to_csv(self.metrics_dir / "scheme_metrics.csv", index=False)

        # 4. Step Marking Metrics CSV
        pd.DataFrame([step_mark_metrics]).T.reset_index().rename(
            columns={"index": "metric", 0: "value"}
        ).to_csv(self.metrics_dir / "step_marking_metrics.csv", index=False)

        logger.info(f"Exported all focused metric CSV files to {self.metrics_dir}")

    def generate_markdown_report(
        self,
        q_metrics: Dict[str, Any],
        step_val_metrics: Dict[str, Any],
        scheme_metrics: Dict[str, Any],
        step_mark_metrics: Dict[str, Any],
        num_questions: int,
        num_steps: int,
    ) -> Path:
        """Generates clear Markdown thesis evaluation report."""
        md_content = f"""# Evaluation Report: AI-Based Stepwise Algebra Evaluation System

**Dataset**: Advanced Level Stepwise Mathematics Grading Benchmark  
**Total Questions ($N_Q$)**: {num_questions} | **Total Solution Steps ($N_S$)**: {num_steps}

---

## 1. Question-Level Evaluation

| Metric | Measured Value | Interpretation |
| :--- | :---: | :--- |
| **MAE** | **{q_metrics['MAE']:.4f}** | Average magnitude of mark error per question. |
| **Pearson Correlation (r)** | **{q_metrics['Pearson Correlation (r)']:.4f}** | Linear alignment between Teacher and AI marks. |
| **Exact Match Accuracy** | **{q_metrics['Exact Match Accuracy (%)']:.2f}%** | Percentage of exact score matches. |
| **Mean Bias Error (MBE)** | **{q_metrics['Mean Bias Error (MBE)']:.4f}** | Directional bias (Positive = Lenient, Negative = Strict). |
| **RMSE** | **{q_metrics['RMSE']:.4f}** | Root mean squared error (penalizes larger errors). |
| **R² Score** | **{q_metrics['R² Score']:.4f}** | Proportion of teacher score variance explained by AI. |

---

## 2. Step-Level Evaluation (Validity Classification)

| Metric | Measured Value | Interpretation |
| :--- | :---: | :--- |
| **Macro F1** | **{step_val_metrics['Macro F1']:.4f}** | Unweighted mean F1-score across all validity classes. |
| **Weighted F1** | **{step_val_metrics['Weighted F1']:.4f}** | Class-weighted F1-score accounting for sample imbalance. |
| **Micro F1** | **{step_val_metrics['Micro F1']:.4f}** | Global F1-score across all individual predictions. |
| **F1 (correct)** | **{step_val_metrics.get('F1 (correct)', 0.0):.4f}** | F1-score for identifying fully valid mathematical steps. |
| **F1 (partially_correct)** | **{step_val_metrics.get('F1 (partially_correct)', 0.0):.4f}** | F1-score for identifying partially correct steps. |
| **F1 (incorrect)** | **{step_val_metrics.get('F1 (incorrect)', 0.0):.4f}** | F1-score for identifying invalid/wrong steps. |
| **Cohen's κ** | **{step_val_metrics["Cohen's κ"]:.4f}** | Inter-rater agreement controlling for chance. |
| **Accuracy** | **{step_val_metrics['Accuracy'] * 100:.2f}%** | Overall step correctness classification rate. |
| **Precision** | **{step_val_metrics['Precision']:.4f}** | Weighted precision across validity classes. |
| **Recall** | **{step_val_metrics['Recall']:.4f}** | Weighted recall across validity classes. |

---

## 3. Step Marking Evaluation

| Metric | Measured Value | Interpretation |
| :--- | :---: | :--- |
| **Step MAE** | **{step_mark_metrics['Step MAE']:.4f}** | Mean absolute error per step mark. |
| **Step RMSE** | **{step_mark_metrics['Step RMSE']:.4f}** | Root mean squared error per step mark. |
| **Pearson r** | **{step_mark_metrics['Pearson r']:.4f}** | Correlation between step-level marks. |
| **Mean Difference** | **{step_mark_metrics['Mean Difference']:.4f}** | Average difference (AI Marks - Teacher Marks). |

---

## 4. Visualizations

### 4.1 System-Assigned Marks vs Human-Assigned Marks (Full Dataset Breakdown)
![System-Assigned Marks vs Human-Assigned Marks](figures/system_vs_human_marks.png)  
*Figure 1: Complete dataset visualization comparing System-Assigned Marks against Human-Assigned Marks (Scatter plot with jitter & test case labels + side-by-side question comparison).*

### 4.2 Error Distributions & Agreement
![Histogram of Absolute Errors](figures/error_histogram.png)  
*Figure 2: Histogram of Question Absolute Errors.*

![Bland-Altman Plot](figures/bland_altman.png)  
*Figure 3: Bland-Altman Plot of System Agreement.*

![Box Plot of Mark Differences](figures/box_plot_errors.png)  
*Figure 4: Box Plot of Mark Differences (Question & Step Level).*

### 4.3 Confusion Matrices & Correlation
![Validity Confusion Matrix](figures/validity_confusion.png)  
*Figure 5: Step Validity Confusion Matrix Heatmap.*

![Scheme Matching Confusion Matrix](figures/scheme_confusion.png)  
*Figure 6: Scheme Matching Confusion Matrix Heatmap.*

![Correlation Heatmap](figures/correlation_heatmap.png)  
*Figure 7: Correlation Heatmap of Evaluation Metrics.*
"""
        self.report_md_path.write_text(md_content, encoding="utf-8")
        logger.info(f"Generated Markdown report at {self.report_md_path}")
        return self.report_md_path

    def generate_pdf_report(
        self,
        q_metrics: Dict[str, Any],
        step_val_metrics: Dict[str, Any],
        step_mark_metrics: Dict[str, Any],
    ) -> Optional[Path]:
        """Compiles clean PDF thesis evaluation report using ReportLab."""
        try:
            from reportlab.lib.pagesizes import letter
            from reportlab.lib import colors
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
        except ImportError:
            logger.warning("ReportLab library not installed. PDF report generation skipped.")
            return None

        doc = SimpleDocTemplate(
            str(self.report_pdf_path),
            pagesize=letter,
            rightMargin=36,
            leftMargin=36,
            topMargin=36,
            bottomMargin=36,
        )

        styles = getSampleStyleSheet()
        title_style = ParagraphStyle("DocTitle", parent=styles["Title"], fontSize=18, leading=22, textColor=colors.HexColor("#2c3e50"), spaceAfter=15)
        h1_style = ParagraphStyle("Heading1_Custom", parent=styles["Heading1"], fontSize=14, leading=18, textColor=colors.HexColor("#2980b9"), spaceBefore=12, spaceAfter=8)
        body_style = ParagraphStyle("Body_Custom", parent=styles["Normal"], fontSize=10, leading=14, textColor=colors.HexColor("#333333"), spaceAfter=6)

        elements = []

        # Title
        elements.append(Paragraph("AI-Based Stepwise Algebra Evaluation Report", title_style))
        elements.append(Paragraph("<b>Benchmark Evaluation Results: Teacher Ground Truth vs AI System</b>", body_style))
        elements.append(Spacer(1, 10))

        # 1. Question Level Metrics Table
        elements.append(Paragraph("1. Question-Level Metrics", h1_style))
        q_data = [
            ["Metric", "Value"],
            ["MAE", f"{q_metrics['MAE']:.4f}"],
            ["Pearson Correlation (r)", f"{q_metrics['Pearson Correlation (r)']:.4f}"],
            ["Exact Match Accuracy", f"{q_metrics['Exact Match Accuracy (%)']:.2f}%"],
            ["Exact Match F1", f"{q_metrics.get('Exact Match F1', 0.0):.4f}"],
            ["Mean Bias Error (MBE)", f"{q_metrics['Mean Bias Error (MBE)']:.4f}"],
            ["RMSE", f"{q_metrics['RMSE']:.4f}"],
            ["R² Score", f"{q_metrics['R² Score']:.4f}"],
        ]
        t_q = Table(q_data, colWidths=[240, 240])
        t_q.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#34495e")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#bdc3c7")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8f9fa")]),
        ]))
        elements.append(t_q)
        elements.append(Spacer(1, 15))

        # 2. Step Validity Table
        elements.append(Paragraph("2. Step-Level Evaluation (Validity Classification)", h1_style))
        step_data = [
            ["Metric", "Value"],
            ["Macro F1", f"{step_val_metrics['Macro F1']:.4f}"],
            ["Weighted F1", f"{step_val_metrics['Weighted F1']:.4f}"],
            ["Micro F1", f"{step_val_metrics['Micro F1']:.4f}"],
            ["F1 (correct)", f"{step_val_metrics.get('F1 (correct)', 0.0):.4f}"],
            ["F1 (partially_correct)", f"{step_val_metrics.get('F1 (partially_correct)', 0.0):.4f}"],
            ["F1 (incorrect)", f"{step_val_metrics.get('F1 (incorrect)', 0.0):.4f}"],
            ["Cohen's κ", f"{step_val_metrics['Cohen\'s κ']:.4f}"],
            ["Accuracy", f"{step_val_metrics['Accuracy'] * 100:.2f}%"],
            ["Precision", f"{step_val_metrics['Precision']:.4f}"],
            ["Recall", f"{step_val_metrics['Recall']:.4f}"],
        ]
        t_step = Table(step_data, colWidths=[240, 240])
        t_step.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2980b9")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#bdc3c7")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8f9fa")]),
        ]))
        elements.append(t_step)
        elements.append(Spacer(1, 15))

        # 3. Step Marking Table
        elements.append(Paragraph("3. Step Marking Evaluation", h1_style))
        mark_data = [
            ["Metric", "Value"],
            ["Step MAE", f"{step_mark_metrics['Step MAE']:.4f}"],
            ["Step RMSE", f"{step_mark_metrics['Step RMSE']:.4f}"],
            ["Pearson r", f"{step_mark_metrics['Pearson r']:.4f}"],
            ["Mean Difference", f"{step_mark_metrics['Mean Difference']:.4f}"],
        ]
        t_mark = Table(mark_data, colWidths=[240, 240])
        t_mark.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#27ae60")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#bdc3c7")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8f9fa")]),
        ]))
        elements.append(t_mark)
        elements.append(Spacer(1, 15))

        # Visualizations
        elements.append(Paragraph("4. Visualizations", h1_style))
        figures_dir = self.report_md_path.parent / "figures"
        img0 = figures_dir / "system_vs_human_marks.png"
        img1 = figures_dir / "error_histogram.png"
        img2 = figures_dir / "bland_altman.png"
        img3 = figures_dir / "box_plot_errors.png"
        img4 = figures_dir / "validity_confusion.png"
        img5 = figures_dir / "scheme_confusion.png"
        img6 = figures_dir / "correlation_heatmap.png"

        for img_path in [img0, img1, img2, img3, img4, img5, img6]:
            if img_path.exists():
                elements.append(Image(str(img_path), width=440, height=240 if img_path == img0 else 260))
                elements.append(Spacer(1, 10))

        try:
            doc.build(elements)
            logger.info(f"Generated PDF report at {self.report_pdf_path}")
            return self.report_pdf_path
        except Exception as err:
            logger.warning(f"Could not overwrite PDF report at {self.report_pdf_path} ({err}). If the file is open in a PDF viewer, please close it.")
            return None
