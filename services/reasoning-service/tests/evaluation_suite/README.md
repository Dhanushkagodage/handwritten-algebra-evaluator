# AI-Based Stepwise Algebra Evaluation System - Research Evaluation Suite

A modular Python framework designed for evaluating AI-based stepwise mathematics grading systems against expert teacher ground truth data.

---

## Output Directory Structure

Executing the evaluation suite creates the following directory structure inside `results/`:

```
results/
├── figures/                     # 15 publication-grade figures (300 DPI)
│   ├── scatter_marks.png        # Scatter Plot: Teacher GT vs AI System Marks (with y=x line)
│   ├── error_histogram.png      # Histogram of Question Absolute Errors
│   ├── box_plot_errors.png      # Box Plot of Mark Differences (Question & Step levels)
│   ├── bland_altman.png         # Bland–Altman Plot (System Agreement & Limits of Agreement)
│   ├── exact_match_dist.png     # Exact Match Breakdown Distribution
│   ├── topic_mae.png            # Topic-wise MAE Bar Chart
│   ├── validity_confusion.png   # Step Validity Confusion Matrix Heatmap
│   ├── scheme_confusion.png     # Scheme Matching Confusion Matrix Heatmap
│   ├── step_mark_diff_hist.png  # Step Mark Difference Histogram
│   ├── validity_distribution.png# Step Validity Prediction Breakdown
│   ├── correlation_heatmap.png  # Metrics Correlation Matrix Heatmap
│   ├── error_by_question.png    # Bar Chart of Errors by Question ID
│   ├── top_10_largest_errors.png# Top 10 Largest Errors Horizontal Bar Chart
│   ├── avg_marks_per_topic.png  # Average Marks per Topic: Teacher GT vs AI System
│   └── exact_match_percentage.png# Overall Exact Match Rate Pie Chart
├── metrics/                     # Exported Quantitative Metrics CSVs
│   ├── question_metrics.csv     # 14 Question-Level Metrics (Exact Match, MAE, RMSE, Pearson r, etc.)
│   ├── step_metrics.csv         # Step Validity Classification Metrics (Acc, Macro F1, Kappa)
│   ├── scheme_metrics.csv       # Scheme Matching Agreement Metrics
│   ├── step_marking_metrics.csv # Step Mark Allocation Errors (MAE, RMSE, Pearson r)
│   ├── topic_metrics.csv        # Topic-wise Metrics Breakdown
│   └── statistical_tests.csv    # Paired t-test, Wilcoxon Test, and Cohen's d Results
├── report.md                    # Research-grade Markdown Thesis Report
└── report.pdf                   # Styled PDF Thesis Evaluation Report
```

---

## Instructions to Execute Manually

Follow these simple steps in your terminal/command prompt to execute the evaluation script manually:

### Step 1: Navigate to the evaluation suite directory
```bash
cd services/reasoning-service/tests/evaluation_suite
```

### Step 2: Install dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Run the evaluation pipeline
```bash
python run_evaluation.py
```

---

## Custom Execution Options

You can also pass custom input CSV files or output directory paths:

```bash
python run_evaluation.py --question-csv "3_output_results/question_level_results.csv" --step-csv "3_output_results/step_level_results.csv" --output-dir "results"
```
