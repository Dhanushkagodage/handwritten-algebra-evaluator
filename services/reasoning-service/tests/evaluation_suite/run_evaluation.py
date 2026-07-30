"""
Main execution script for the AI-Based Stepwise Algebra Evaluation System Analysis Suite.

Usage:
    python run_evaluation.py [--question-csv PATH] [--step-csv PATH] [--output-dir PATH]
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import QUESTION_LEVEL_CSV, STEP_LEVEL_CSV, DEFAULT_OUTPUT_DIR
from evaluator.data_loader import DataLoader
from evaluator.metrics_calculator import MetricsCalculator
from evaluator.visualizer import Visualizer
from evaluator.report_generator import ReportGenerator

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("RunEvaluation")


def print_cli_summary(
    q_metrics: Dict[str, Any],
    step_val_metrics: Dict[str, Any],
    step_mark_metrics: Dict[str, Any],
    num_questions: int,
    num_steps: int,
) -> None:
    """Prints a beautiful, structured ASCII table summary directly to the terminal."""
    border_line = "=" * 90
    section_line = "-" * 90

    print("\n" + border_line)
    print("         AI-BASED STEPWISE ALGEBRA EVALUATION SYSTEM — FINAL BENCHMARK SUMMARY")
    print(f"         Total Questions Evaluated: {num_questions}  |  Total Solution Steps Evaluated: {num_steps}")
    print(border_line)

    # 1. Question-Level Metrics
    print("\n" + section_line)
    print("  1. QUESTION-LEVEL EVALUATION METRICS (TOTAL MARKS AGREEMENT)")
    print(section_line)
    print(f"  {'Metric Name':<42} | {'Measured Value':<20} | {'Benchmark Standard':<20}")
    print(section_line)
    print(f"  {'MAE (Mean Absolute Error)':<42} | {q_metrics['MAE']:<20.4f} | {'< 0.50 marks':<20}")
    print(f"  {'Pearson Correlation (r)':<42} | {q_metrics['Pearson Correlation (r)']:<20.4f} | {'> 0.90':<20}")
    exact_acc_str = f"{q_metrics['Exact Match Accuracy (%)']:.2f}%"
    print(f"  {'Exact Match Accuracy (%)':<42} | {exact_acc_str:<20} | {'> 80.0%':<20}")
    print(f"  {'Exact Match F1 Score':<42} | {q_metrics.get('Exact Match F1', 0.0):<20.4f} | {'> 0.80':<20}")
    print(f"  {'Mean Bias Error (MBE)':<42} | {q_metrics['Mean Bias Error (MBE)']:<20.4f} | {'~ 0.00':<20}")
    print(f"  {'RMSE (Root Mean Squared Error)':<42} | {q_metrics['RMSE']:<20.4f} | {'< 1.00 marks':<20}")
    print(f"  {'R² Score':<42} | {q_metrics['R² Score']:<20.4f} | {'> 0.85':<20}")
    print(section_line)

    # 2. Step-Level Validity & F1 Metrics
    print("\n" + section_line)
    print("  2. STEP-LEVEL EVALUATION (VALIDITY CLASSIFICATION & F1 BREAKDOWN)")
    print(section_line)
    print(f"  {'Metric Name':<42} | {'Measured Value':<20} | {'Interpretation':<20}")
    print(section_line)
    print(f"  {'Macro F1 Score':<42} | {step_val_metrics['Macro F1']:<20.4f} | {'Unweighted Class F1':<20}")
    print(f"  {'Weighted F1 Score':<42} | {step_val_metrics['Weighted F1']:<20.4f} | {'Class-Weighted F1':<20}")
    print(f"  {'Micro F1 Score':<42} | {step_val_metrics['Micro F1']:<20.4f} | {'Global Step F1':<20}")
    print(f"  {'F1 Score (correct)':<42} | {step_val_metrics.get('F1 (correct)', 0.0):<20.4f} | {'Valid Step Accuracy':<20}")
    print(f"  {'F1 Score (partially_correct)':<42} | {step_val_metrics.get('F1 (partially_correct)', 0.0):<20.4f} | {'Partial Step Accuracy':<20}")
    print(f"  {'F1 Score (incorrect)':<42} | {step_val_metrics.get('F1 (incorrect)', 0.0):<20.4f} | {'Error Detection F1':<20}")
    print(f"  {'Cohen\'s Kappa (κ)':<42} | {step_val_metrics['Cohen\'s κ']:<20.4f} | {'Substantial (>0.80)':<20}")
    step_acc_str = f"{step_val_metrics['Accuracy'] * 100:.2f}%"
    print(f"  {'Accuracy (%)':<42} | {step_acc_str:<20} | {'Overall Correctness':<20}")
    print(f"  {'Precision':<42} | {step_val_metrics['Precision']:<20.4f} | {'Weighted Precision':<20}")
    print(f"  {'Recall':<42} | {step_val_metrics['Recall']:<20.4f} | {'Weighted Recall':<20}")
    print(section_line)

    # 3. Step Marking Metrics
    print("\n" + section_line)
    print("  3. STEP MARKING EVALUATION METRICS (MARK ALLOCATION PRECISION)")
    print(section_line)
    print(f"  {'Metric Name':<42} | {'Measured Value':<20} | {'Interpretation':<20}")
    print(section_line)
    print(f"  {'Step MAE':<42} | {step_mark_metrics['Step MAE']:<20.4f} | {'Mean mark error/step':<20}")
    print(f"  {'Step RMSE':<42} | {step_mark_metrics['Step RMSE']:<20.4f} | {'Root mean sq error':<20}")
    print(f"  {'Pearson r':<42} | {step_mark_metrics['Pearson r']:<20.4f} | {'Step mark correlation':<20}")
    print(f"  {'Mean Difference':<42} | {step_mark_metrics['Mean Difference']:<20.4f} | {'Mean mark offset':<20}")
    print(section_line)
    print("\n" + border_line)
    print("  ALL RESULTS & FIGURES EXPORTED TO 'results/' DIRECTORY FOR THESIS INCLUSION")
    print(border_line + "\n")


def main():
    parser = argparse.ArgumentParser(description="Run Evaluation Suite for Stepwise Algebra AI System")
    parser.add_argument("--question-csv", type=str, default=str(QUESTION_LEVEL_CSV), help="Path to question_level_results.csv")
    parser.add_argument("--step-csv", type=str, default=str(STEP_LEVEL_CSV), help="Path to step_level_results.csv")
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR), help="Output directory")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    figures_dir = output_dir / "figures"
    metrics_dir = output_dir / "metrics"
    report_md = output_dir / "report.md"
    report_pdf = output_dir / "report.pdf"

    logger.info("================================================================================")
    logger.info("  AI-Based Stepwise Algebra Evaluation System - Research Evaluation Suite")
    logger.info("================================================================================")

    # 1. Load Data
    logger.info("Step 1/4: Loading and preprocessing datasets...")
    loader = DataLoader(question_csv=Path(args.question_csv), step_csv=Path(args.step_csv))
    df_question, df_step = loader.load_data()
    logger.info(f"Loaded {len(df_question)} question test cases and {len(df_step)} solution steps.")

    # 2. Compute Metrics
    logger.info("Step 2/4: Calculating evaluation metrics...")
    calc = MetricsCalculator(df_question, df_step)
    q_metrics = calc.compute_question_metrics()
    step_val_metrics = calc.compute_step_validity_metrics()
    scheme_metrics = calc.compute_scheme_matching_metrics()
    step_mark_metrics = calc.compute_step_marking_metrics()

    # 3. Generate Visualizations (300 DPI)
    logger.info("Step 3/4: Generating 6 plot-based figures (300 DPI)...")
    viz = Visualizer(output_dir=figures_dir)
    viz.generate_all_figures(
        df_question=df_question,
        df_step=df_step,
        validity_cm=step_val_metrics["confusion_matrix"],
        validity_labels=step_val_metrics["labels"],
        scheme_cm=scheme_metrics["confusion_matrix"],
        scheme_labels=scheme_metrics["labels"],
    )

    # 4. Export Metrics & Compile Reports
    logger.info("Step 4/4: Exporting metric CSV files and compiling Markdown & PDF thesis reports...")
    report_gen = ReportGenerator(
        metrics_dir=metrics_dir,
        report_md_path=report_md,
        report_pdf_path=report_pdf,
    )
    report_gen.export_csv_metrics(
        q_metrics=q_metrics,
        step_val_metrics=step_val_metrics,
        scheme_metrics=scheme_metrics,
        step_mark_metrics=step_mark_metrics,
    )
    report_gen.generate_markdown_report(
        q_metrics=q_metrics,
        step_val_metrics=step_val_metrics,
        scheme_metrics=scheme_metrics,
        step_mark_metrics=step_mark_metrics,
        num_questions=len(df_question),
        num_steps=len(df_step),
    )
    report_gen.generate_pdf_report(
        q_metrics=q_metrics,
        step_val_metrics=step_val_metrics,
        step_mark_metrics=step_mark_metrics,
    )

    # 5. Print CLI Terminal Summary Table for Screenshots
    print_cli_summary(
        q_metrics=q_metrics,
        step_val_metrics=step_val_metrics,
        step_mark_metrics=step_mark_metrics,
        num_questions=len(df_question),
        num_steps=len(df_step),
    )


if __name__ == "__main__":
    main()
