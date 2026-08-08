"""
csv_recorder.py (Single LLM Baseline)
──────────────────────────────────────
Appends Single LLM evaluation results + ground-truth data to two CSV files
located in tests/evaluation_suite/3_output_results/ after every
successful single-LLM evaluation call that provides a test_case_id.

CSV files:
  single_llm_question_level_results.csv  — one row per test-case
  single_llm_step_level_results.csv      — one row per student step per test-case
"""
import csv
import json
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────

_SUITE_ROOT = (
    Path(__file__).resolve()
    .parent.parent.parent
    / "tests" / "evaluation_suite"
)

GT_DIR      = _SUITE_ROOT / "2_ground_truth"
RESULTS_DIR = _SUITE_ROOT / "3_output_results"

SINGLE_Q_CSV = RESULTS_DIR / "single_llm_question_level_results.csv"
SINGLE_S_CSV = RESULTS_DIR / "single_llm_step_level_results.csv"

# ── Column headers (matches multi-agent CSV structure) ─────────────────────────

Q_HEADERS = [
    "test_case_id",
    "question_text",
    "max_marks",
    "gt_total_marks",
    "sys_total_marks",
    "absolute_error",
    "squared_error",
    "exact_match",
    "detected_method",
    "summary_feedback",
]

S_HEADERS = [
    "test_case_id",
    "step_id",
    "gt_matched_scheme",
    "sys_matched_scheme",
    "scheme_match_agree",
    "gt_validity",
    "sys_validity",
    "validity_agree",
    "gt_marks",
    "sys_marks",
    "mark_diff",
]


def _ensure_csv(path: Path, headers: list[str]) -> None:
    """Create the CSV with headers if it does not exist or is empty."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists() or path.stat().st_size == 0:
        with open(path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(headers)
        logger.info("[single_llm.csv_recorder] Initialised headers in %s", path.name)


def _load_ground_truth(test_case_id: str) -> Optional[dict]:
    """Load tc_NNN_gt.json; return None if missing."""
    gt_path = GT_DIR / f"{test_case_id}_gt.json"
    if not gt_path.exists():
        logger.warning("[single_llm.csv_recorder] Ground-truth not found: %s", gt_path)
        return None
    with open(gt_path, "r", encoding="utf-8") as f:
        return json.load(f)


def record_single_llm_evaluation(
    test_case_id: str,
    question_text: str,
    max_marks: float,
    api_output: dict,
) -> dict:
    """
    Load ground truth for test_case_id, then append rows to single LLM CSVs.
    """
    _ensure_csv(SINGLE_Q_CSV, Q_HEADERS)
    _ensure_csv(SINGLE_S_CSV, S_HEADERS)

    gt = _load_ground_truth(test_case_id)
    if gt is None:
        logger.warning(
            "[single_llm.csv_recorder] Skipping CSV write for %s — no ground truth found.",
            test_case_id,
        )
        return {
            "gt_loaded": False,
            "question_row_written": False,
            "step_rows_written": 0,
        }

    # ── Question-level row ────────────────────────────────────────────────────
    gt_total  = float(gt.get("gt_total_marks", 0))
    sys_total = float(api_output.get("total_marks", 0))
    abs_err   = abs(gt_total - sys_total)
    sq_err    = (gt_total - sys_total) ** 2
    exact     = gt_total == sys_total

    # Extract method from first step or summary feedback
    steps_analysis = api_output.get("steps_analysis", [])
    detected_method = ""
    if steps_analysis and isinstance(steps_analysis, list) and len(steps_analysis) > 0:
        detected_method = steps_analysis[0].get("method", "")

    q_row = {
        "test_case_id":   test_case_id,
        "question_text":  question_text,
        "max_marks":      max_marks,
        "gt_total_marks": gt_total,
        "sys_total_marks": sys_total,
        "absolute_error": round(abs_err, 4),
        "squared_error":  round(sq_err, 4),
        "exact_match":    exact,
        "detected_method": detected_method,
        "summary_feedback": api_output.get("summary", ""),
    }

    with open(SINGLE_Q_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=Q_HEADERS)
        writer.writerow(q_row)

    logger.info(
        "[single_llm.csv_recorder] Single LLM Question row written — %s | gt=%.1f sys=%.1f err=%.1f",
        test_case_id, gt_total, sys_total, abs_err,
    )

    # ── Step-level rows ───────────────────────────────────────────────────────
    gt_steps  = {s["step_id"]: s for s in gt.get("gt_steps_analysis", [])}
    sys_steps = {s["step_id"]: s for s in steps_analysis}

    step_rows_written = 0
    s_rows: list[dict] = []

    for sid in sorted(gt_steps.keys()):
        gt_s  = gt_steps.get(sid, {})
        sys_s = sys_steps.get(sid, {})

        gt_scheme  = gt_s.get("matched_scheme_step")
        sys_scheme = sys_s.get("matched_scheme_step")
        gt_val     = gt_s.get("validity", "")
        sys_val    = sys_s.get("status", "")
        gt_m       = float(gt_s.get("gt_marks_awarded", 0))
        sys_m      = float(sys_s.get("marks_awarded", 0)) if sys_s else 0.0

        s_rows.append({
            "test_case_id":       test_case_id,
            "step_id":            sid,
            "gt_matched_scheme":  gt_scheme,
            "sys_matched_scheme": sys_scheme,
            "scheme_match_agree": gt_scheme == sys_scheme,
            "gt_validity":        gt_val,
            "sys_validity":       sys_val,
            "validity_agree":     gt_val == sys_val,
            "gt_marks":           gt_m,
            "sys_marks":          sys_m,
            "mark_diff":          round(gt_m - sys_m, 4),
        })
        step_rows_written += 1

    with open(SINGLE_S_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=S_HEADERS)
        writer.writerows(s_rows)

    logger.info(
        "[single_llm.csv_recorder] Single LLM Step rows written — %s | %d steps",
        test_case_id, step_rows_written,
    )

    return {
        "gt_loaded":             True,
        "question_row_written":  True,
        "step_rows_written":     step_rows_written,
        "csv_paths": {
            "question_level": str(SINGLE_Q_CSV),
            "step_level":     str(SINGLE_S_CSV),
        },
    }
