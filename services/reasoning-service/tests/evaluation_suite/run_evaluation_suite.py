"""
Evaluation Suite Runner Script for Reasoning Service
────────────────────────────────────────────────────
Executes system evaluation across all test cases in 1_inputs/, compares with 2_ground_truth/,
computes quantitative statistical metrics (MAE, RMSE, Exact Match %, Pearson r),
and saves CSV results into 3_output_results/.
"""

import sys
import os
import glob
import json
import csv
import math
import asyncio
from pathlib import Path
import urllib.request
import urllib.error

# Ensure root workspace directory is in sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
SERVICE_DIR = SCRIPT_DIR.parent.parent
if str(SERVICE_DIR) not in sys.path:
    sys.path.insert(0, str(SERVICE_DIR))

# Import schemas and pipeline runner from application
try:
    from app.schemas.input_schema import EvaluationRequest
    from app.api.routes import _run_pipeline
    PIPELINE_IMPORT_SUCCESS = True
except Exception as e:
    PIPELINE_IMPORT_SUCCESS = False
    print(f"[Warning] Could not import internal pipeline directly: {e}")

API_URL = "http://localhost:8002/api/v1/evaluate"


def calculate_pearson_r(x_list: list[float], y_list: list[float]) -> float:
    """Compute Pearson correlation coefficient r between two lists of floats."""
    n = len(x_list)
    if n < 2:
        return 0.0
    mean_x = sum(x_list) / n
    mean_y = sum(y_list) / n

    cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(x_list, y_list))
    var_x = sum((x - mean_x) ** 2 for x in x_list)
    var_y = sum((y - mean_y) ** 2 for y in y_list)

    if var_x == 0 or var_y == 0:
        return 0.0

    return cov / (math.sqrt(var_x) * math.sqrt(var_y))


async def get_system_evaluation(input_data: dict) -> dict:
    """
    Evaluates an input payload using either the live HTTP API or direct Python pipeline invocation.
    """
    json_bytes = json.dumps(input_data).encode("utf-8")
    
    # 1. Try live HTTP request first
    try:
        req = urllib.request.Request(
            API_URL,
            data=json_bytes,
            headers={"Content-Type": "application/json"},
            method="POST"
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            if resp.status == 200:
                res_body = resp.read().decode("utf-8")
                return json.loads(res_body)
    except Exception:
        pass  # Fall back to direct Python invocation

    # 2. Direct Python pipeline fallback
    if PIPELINE_IMPORT_SUCCESS:
        request_obj = EvaluationRequest.model_validate(input_data)
        output, state, result = await _run_pipeline(request_obj)
        return output
    else:
        raise RuntimeError("Could not connect to API server and direct Python import failed.")


async def main():
    inputs_dir = SCRIPT_DIR / "1_inputs"
    gt_dir = SCRIPT_DIR / "2_ground_truth"
    out_dir = SCRIPT_DIR / "3_output_results"
    out_dir.mkdir(parents=True, exist_ok=True)

    input_files = sorted(glob.glob(str(inputs_dir / "tc_*.json")))
    if not input_files:
        print(f"Error: No test cases found in {inputs_dir}")
        return

    print(f"Found {len(input_files)} test cases in inputs folder.")
    print("Running evaluation suite...\n" + "─" * 60)

    question_results = []
    step_results = []

    for input_file_path in input_files:
        filename = Path(input_file_path).name
        tc_id = filename.replace(".json", "")
        gt_file_path = gt_dir / f"{tc_id}_gt.json"

        if not gt_file_path.exists():
            print(f"[Warning] Ground truth file missing for {tc_id}: {gt_file_path}")
            continue

        with open(input_file_path, "r", encoding="utf-8") as f:
            input_json = json.load(f)

        with open(gt_file_path, "r", encoding="utf-8") as f:
            gt_json = json.load(f)

        print(f"Evaluating [{tc_id}] ... ", end="", flush=True)

        try:
            sys_output = await get_system_evaluation(input_json)
            print("DONE")
        except Exception as err:
            print(f"FAILED ({err})")
            continue

        # Extract Question Level Data
        max_marks = float(gt_json.get("total_available_marks", input_json["marking_scheme"]["total_marks"]))
        gt_total = float(gt_json.get("gt_total_marks", 0.0))
        sys_total = float(sys_output.get("total_marks", 0.0))

        abs_err = abs(gt_total - sys_total)
        sq_err = (gt_total - sys_total) ** 2
        exact_match = (gt_total == sys_total)
        detected_method = sys_output.get("method_detection", {}).get("detected_method", "undetermined")
        if isinstance(sys_output.get("method_detection"), dict):
            detected_method = sys_output["method_detection"].get("detected_method", "undetermined")
        summary_feedback = sys_output.get("summary", "")

        q_row = {
            "test_case_id": tc_id,
            "question_text": input_json["reasoning_input"]["question_text"],
            "max_marks": max_marks,
            "gt_total_marks": gt_total,
            "sys_total_marks": sys_total,
            "absolute_error": round(abs_err, 2),
            "squared_error": round(sq_err, 4),
            "exact_match": exact_match,
            "detected_method": detected_method,
            "summary_feedback": summary_feedback
        }
        question_results.append(q_row)

        # Extract Step Level Data
        gt_steps_map = {s["step_id"]: s for s in gt_json.get("gt_steps_analysis", [])}
        sys_steps_list = sys_output.get("steps_analysis", [])

        for sys_step in sys_steps_list:
            s_id = sys_step["step_id"]
            gt_step = gt_steps_map.get(s_id, {})

            gt_matched = gt_step.get("matched_scheme_step")
            sys_matched = sys_step.get("matched_scheme_step")
            scheme_match_agree = (gt_matched == sys_matched)

            gt_validity = gt_step.get("validity", "unclear")
            sys_validity = sys_step.get("status", "unclear")
            validity_agree = (gt_validity == sys_validity)

            gt_m = float(gt_step.get("gt_marks_awarded", 0.0))
            sys_m = float(sys_step.get("marks_awarded", 0.0))
            m_diff = abs(gt_m - sys_m)

            s_row = {
                "test_case_id": tc_id,
                "step_id": s_id,
                "gt_matched_scheme": gt_matched if gt_matched is not None else "",
                "sys_matched_scheme": sys_matched if sys_matched is not None else "",
                "scheme_match_agree": scheme_match_agree,
                "gt_validity": gt_validity,
                "sys_validity": sys_validity,
                "validity_agree": validity_agree,
                "gt_marks": gt_m,
                "sys_marks": sys_m,
                "mark_diff": round(m_diff, 2)
            }
            step_results.append(s_row)

    # ── Write Question Level CSV ──────────────────────────────────────────────
    q_csv_path = out_dir / "question_level_results.csv"
    if question_results:
        q_headers = list(question_results[0].keys())
        with open(q_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=q_headers)
            writer.writeheader()
            writer.writerows(question_results)

    # ── Write Step Level CSV ──────────────────────────────────────────────────
    s_csv_path = out_dir / "step_level_results.csv"
    if step_results:
        s_headers = list(step_results[0].keys())
        with open(s_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=s_headers)
            writer.writeheader()
            writer.writerows(step_results)

    # ── Compute Summary Quantitative Metrics ──────────────────────────────────
    print("\n" + "=" * 60)
    print("EVALUATION SUITE METRIC RESULTS SUMMARY")
    print("=" * 60)

    if question_results:
        gt_totals = [q["gt_total_marks"] for q in question_results]
        sys_totals = [q["sys_total_marks"] for q in question_results]
        abs_errors = [q["absolute_error"] for q in question_results]
        sq_errors = [q["squared_error"] for q in question_results]
        exact_matches = [1 if q["exact_match"] else 0 for q in question_results]

        mae = sum(abs_errors) / len(abs_errors)
        rmse = math.sqrt(sum(sq_errors) / len(sq_errors))
        exact_acc = (sum(exact_matches) / len(exact_matches)) * 100.0
        pearson_r = calculate_pearson_r(gt_totals, sys_totals)

        print(f"Questions Evaluated     : {len(question_results)}")
        print(f"Question Total MAE      : {mae:.2f} marks")
        print(f"Question Total RMSE     : {rmse:.2f} marks")
        print(f"Question Exact Match %  : {exact_acc:.1f}%")
        print(f"Pearson Correlation (r) : {pearson_r:.4f}")

    if step_results:
        step_diffs = [s["mark_diff"] for s in step_results]
        validity_matches = [1 if s["validity_agree"] else 0 for s in step_results]
        scheme_matches = [1 if s["scheme_match_agree"] else 0 for s in step_results]
        step_exact_matches = [1 if s["mark_diff"] == 0 else 0 for s in step_results]

        step_mae = sum(step_diffs) / len(step_diffs)
        val_acc = (sum(validity_matches) / len(validity_matches)) * 100.0
        scheme_acc = (sum(scheme_matches) / len(scheme_matches)) * 100.0
        step_exact_acc = (sum(step_exact_matches) / len(step_exact_matches)) * 100.0

        print("─" * 60)
        print(f"Total Solution Steps    : {len(step_results)}")
        print(f"Per-Step Mark MAE       : {step_mae:.2f} marks")
        print(f"Step Exact Mark Match % : {step_exact_acc:.1f}%")
        print(f"Validity Agreement %    : {val_acc:.1f}%")
        print(f"Scheme Matching Acc %   : {scheme_acc:.1f}%")

    print("\nResults exported successfully:")
    print(f" - {q_csv_path}")
    print(f" - {s_csv_path}")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
