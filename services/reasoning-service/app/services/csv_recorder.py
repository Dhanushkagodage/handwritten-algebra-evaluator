"""
csv_recorder.py
───────────────
Appends evaluation results + ground-truth data to the two CSV files
located in tests/evaluation_suite/3_output_results/ after every
successful /api/v1/evaluate call that provides a test_case_id.

CSV files:
  question_level_results.csv  — one row per test-case
  step_level_results.csv      — one row per student step per test-case
"""
import csv
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────

_SUITE_ROOT = (
    Path(__file__).resolve()              # .../app/services/csv_recorder.py
    .parent                               # .../app/services/
    .parent                               # .../app/
    .parent                               # .../reasoning-service/
    / "tests" / "evaluation_suite"
)

GT_DIR      = _SUITE_ROOT / "2_ground_truth"
RESULTS_DIR = _SUITE_ROOT / "3_output_results"

Q_CSV   = RESULTS_DIR / "question_level_results.csv"
S_CSV   = RESULTS_DIR / "step_level_results.csv"

# ── Column headers (must match existing CSVs) ─────────────────────────────────

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


# ── Internal helpers ──────────────────────────────────────────────────────────

def _ensure_csv(path: Path, headers: list[str]) -> None:
    """Create the CSV with headers if it does not exist or is empty."""
    path.parent.mkdir(parents=True, exist_ok=True)
    # Write headers when file is missing OR empty (e.g. user cleared it)
    if not path.exists() or path.stat().st_size == 0:
        with open(path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(headers)
        logger.info("[csv_recorder] Initialised headers in %s", path.name)


def _load_ground_truth(test_case_id: str) -> Optional[dict]:
    """Load tc_NNN_gt.json; return None if missing."""
    gt_path = GT_DIR / f"{test_case_id}_gt.json"
    if not gt_path.exists():
        logger.warning("[csv_recorder] Ground-truth not found: %s", gt_path)
        return None
    import json
    with open(gt_path, "r", encoding="utf-8") as f:
        return json.load(f)


# ── Public API ────────────────────────────────────────────────────────────────

def record_evaluation(
    test_case_id: str,
    question_text: str,
    max_marks: float,
    api_output: dict,
) -> dict:
    """
    Load ground truth for *test_case_id*, then append rows to both CSVs.

    Parameters
    ----------
    test_case_id : str
        e.g. "tc_001"
    question_text : str
        The original question text (from the request body).
    max_marks : float
        Total marks available (marking_scheme.total_marks).
    api_output : dict
        The raw dict returned by _run_pipeline() — same dict used to build
        EvaluationOutput.  Must contain:
          - total_marks          (float)
          - steps_analysis       (list of dicts with step_id, matched_scheme_step,
                                  status, marks_awarded)
          - method_detection     (dict with detected_method, may be None)
          - summary              (str)

    Returns
    -------
    dict  — record_summary with gt_loaded, rows written, csv paths.
    """
    _ensure_csv(Q_CSV, Q_HEADERS)
    _ensure_csv(S_CSV, S_HEADERS)

    gt = _load_ground_truth(test_case_id)

    if gt is None:
        logger.warning(
            "[csv_recorder] Skipping CSV write for %s — no ground truth found.",
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

    method_info     = api_output.get("method_detection") or {}
    detected_method = method_info.get("detected_method", "")

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

    with open(Q_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=Q_HEADERS)
        writer.writerow(q_row)

    logger.info(
        "[csv_recorder] Question row written — %s | gt=%.1f sys=%.1f err=%.1f",
        test_case_id, gt_total, sys_total, abs_err,
    )

    # ── Step-level rows ───────────────────────────────────────────────────────
    # Build lookup dicts keyed by step_id
    gt_steps  = {s["step_id"]: s for s in gt.get("gt_steps_analysis", [])}
    sys_steps = {s["step_id"]: s for s in api_output.get("steps_analysis", [])}

    # Iterate over all step_ids present in ground truth
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

    with open(S_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=S_HEADERS)
        writer.writerows(s_rows)

    logger.info(
        "[csv_recorder] Step rows written — %s | %d steps",
        test_case_id, step_rows_written,
    )

    return {
        "gt_loaded":             True,
        "question_row_written":  True,
        "step_rows_written":     step_rows_written,
        "csv_paths": {
            "question_level": str(Q_CSV),
            "step_level":     str(S_CSV),
        },
    }
