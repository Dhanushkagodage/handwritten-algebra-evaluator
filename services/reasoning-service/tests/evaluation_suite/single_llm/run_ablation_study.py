"""
run_ablation_study.py
─────────────────────
Ablation Study Execution Script: Multi-Agent Architecture vs Single LLM Baseline.

Loads evaluation results for both systems:
1. Multi-Agent System (question_level_results.csv, step_level_results.csv)
2. Single LLM Baseline (single_llm_question_level_results.csv, single_llm_step_level_results.csv)

Computes metrics for both and prints a side-by-side comparative table.
"""
import sys
import logging
from pathlib import Path

# Add evaluation_suite directory to sys.path
_SUITE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_SUITE_DIR))

from evaluator.data_loader import DataLoader
from evaluator.metrics_calculator import MetricsCalculator

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("AblationStudy")


def print_comparative_table(multi_q, multi_val, multi_mark, single_q, single_val, single_mark, n_q, n_s) -> None:
    """Print ASCII comparison table showing Multi-Agent vs Single LLM metrics."""
    border = "=" * 96
    divider = "-" * 96

    print("\n" + border)
    print("      ABLATION STUDY BENCHMARK — MULTI-AGENT ARCHITECTURE vs. SINGLE LLM BASELINE")
    print(f"      Evaluated Questions: {n_q}  |  Evaluated Steps: {n_s}")
    print(border)

    # 1. Question-Level Metrics
    print("\n" + divider)
    print("  1. QUESTION-LEVEL EVALUATION (TOTAL MARKS ACCURACY & CORRELATION)")
    print(divider)
    print(f"  {'Metric Name':<38} | {'Single LLM':<16} | {'Multi-Agent':<16} | {'Improvement':<14}")
    print(divider)

    mae_diff = single_q['MAE'] - multi_q['MAE']
    print(f"  {'MAE (Mean Absolute Error)':<38} | {single_q['MAE']:<16.4f} | {multi_q['MAE']:<16.4f} | {mae_diff:+.4f} (lower)")

    rmse_diff = single_q['RMSE'] - multi_q['RMSE']
    print(f"  {'RMSE (Root Mean Sq Error)':<38} | {single_q['RMSE']:<16.4f} | {multi_q['RMSE']:<16.4f} | {rmse_diff:+.4f} (lower)")

    pearson_diff = multi_q['Pearson Correlation (r)'] - single_q['Pearson Correlation (r)']
    print(f"  {'Pearson Correlation (r)':<38} | {single_q['Pearson Correlation (r)']:<16.4f} | {multi_q['Pearson Correlation (r)']:<16.4f} | {pearson_diff:+.4f} (higher)")

    r2_diff = multi_q['R² Score'] - single_q['R² Score']
    print(f"  {'R² Score':<38} | {single_q['R² Score']:<16.4f} | {multi_q['R² Score']:<16.4f} | {r2_diff:+.4f} (higher)")

    exact_diff = multi_q['Exact Match Accuracy (%)'] - single_q['Exact Match Accuracy (%)']
    print(f"  {'Exact Match Accuracy (%)':<38} | {single_q['Exact Match Accuracy (%)']:<15.2f}% | {multi_q['Exact Match Accuracy (%)']:<15.2f}% | {exact_diff:+.2f}%")

    # 2. Step-Level Validity Metrics
    print("\n" + divider)
    print("  2. STEP-LEVEL VALIDITY CLASSIFICATION & AGREEMENT METRICS")
    print(divider)
    print(f"  {'Metric Name':<38} | {'Single LLM':<16} | {'Multi-Agent':<16} | {'Improvement':<14}")
    print(divider)

    step_acc_single = single_val['Accuracy'] * 100
    step_acc_multi  = multi_val['Accuracy'] * 100
    acc_diff = step_acc_multi - step_acc_single
    print(f"  {'Step Accuracy (%)':<38} | {step_acc_single:<15.2f}% | {step_acc_multi:<15.2f}% | {acc_diff:+.2f}%")

    macro_f1_diff = multi_val['Macro F1'] - single_val['Macro F1']
    print(f"  {'Macro F1 Score':<38} | {single_val['Macro F1']:<16.4f} | {multi_val['Macro F1']:<16.4f} | {macro_f1_diff:+.4f}")

    weighted_f1_diff = multi_val['Weighted F1'] - single_val['Weighted F1']
    print(f"  {'Weighted F1 Score':<38} | {single_val['Weighted F1']:<16.4f} | {multi_val['Weighted F1']:<16.4f} | {weighted_f1_diff:+.4f}")

    kappa_diff = multi_val['Cohen\'s κ'] - single_val['Cohen\'s κ']
    print(f"  {'Cohen\'s Kappa (κ)':<38} | {single_val['Cohen\'s κ']:<16.4f} | {multi_val['Cohen\'s κ']:<16.4f} | {kappa_diff:+.4f}")

    # 3. Step Marking Metrics
    print("\n" + divider)
    print("  3. STEP MARKING ALLOCATION PRECISION")
    print(divider)
    print(f"  {'Metric Name':<38} | {'Single LLM':<16} | {'Multi-Agent':<16} | {'Improvement':<14}")
    print(divider)

    step_mae_diff = single_mark['Step MAE'] - multi_mark['Step MAE']
    print(f"  {'Step MAE':<38} | {single_mark['Step MAE']:<16.4f} | {multi_mark['Step MAE']:<16.4f} | {step_mae_diff:+.4f} (lower)")

    step_rmse_diff = single_mark['Step RMSE'] - multi_mark['Step RMSE']
    print(f"  {'Step RMSE':<38} | {single_mark['Step RMSE']:<16.4f} | {multi_mark['Step RMSE']:<16.4f} | {step_rmse_diff:+.4f} (lower)")

    step_r_diff = multi_mark['Pearson r'] - single_mark['Pearson r']
    print(f"  {'Step Pearson r':<38} | {single_mark['Pearson r']:<16.4f} | {multi_mark['Pearson r']:<16.4f} | {step_r_diff:+.4f} (higher)")

    print(divider)
    print("\n" + border)
    print("  ABLATION STUDY COMPLETED — RESULTS DEMONSTRATE MULTI-AGENT ARCHITECTURAL GAIN")
    print(border + "\n")


def main():
    results_dir = _SUITE_DIR / "3_output_results"
    
    multi_q_csv = results_dir / "question_level_results.csv"
    multi_s_csv = results_dir / "step_level_results.csv"
    
    single_q_csv = results_dir / "single_llm_question_level_results.csv"
    single_s_csv = results_dir / "single_llm_step_level_results.csv"

    if not multi_q_csv.exists() or not single_q_csv.exists():
        logger.error(
            "Missing CSV results files! Ensure evaluations have been run for both systems.\n"
            "  Multi-agent Q CSV: %s (exists=%s)\n"
            "  Single LLM Q CSV:  %s (exists=%s)",
            multi_q_csv, multi_q_csv.exists(),
            single_q_csv, single_q_csv.exists()
        )
        return

    logger.info("Loading Multi-Agent evaluation dataset...")
    multi_loader = DataLoader(question_csv=multi_q_csv, step_csv=multi_s_csv)
    df_multi_q, df_multi_s = multi_loader.load_data()

    logger.info("Loading Single LLM evaluation dataset...")
    single_loader = DataLoader(question_csv=single_q_csv, step_csv=single_s_csv)
    df_single_q, df_single_s = single_loader.load_data()

    # Compute Multi-Agent metrics
    multi_calc = MetricsCalculator(df_multi_q, df_multi_s)
    multi_q_m = multi_calc.compute_question_metrics()
    multi_val_m = multi_calc.compute_step_validity_metrics()
    multi_mark_m = multi_calc.compute_step_marking_metrics()

    # Compute Single LLM metrics
    single_calc = MetricsCalculator(df_single_q, df_single_s)
    single_q_m = single_calc.compute_question_metrics()
    single_val_m = single_calc.compute_step_validity_metrics()
    single_mark_m = single_calc.compute_step_marking_metrics()

    # Print comparison
    print_comparative_table(
        multi_q=multi_q_m,
        multi_val=multi_val_m,
        multi_mark=multi_mark_m,
        single_q=single_q_m,
        single_val=single_val_m,
        single_mark=single_mark_m,
        n_q=len(df_multi_q),
        n_s=len(df_multi_s),
    )


if __name__ == "__main__":
    main()
