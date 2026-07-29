"""
Validation gate for the feedback-generation training dataset.

Checks raw_annotations.json (and, if present, the regenerated train/eval
splits) against the project's schema, enum vocabularies, marking-consistency
rules, and duplicate/leakage requirements. Exits non-zero with a clear error
list on any HARD failure; distribution/coverage imbalance is reported but
only fails the run if --strict-balance is passed, since hitting an exact
target percentage on every batch during incremental generation is not
realistic — the final 500-example run should be checked with --strict-balance.

Usage (from services/feedback-service):
    python -m app.training.validate_dataset
    python -m app.training.validate_dataset --expected-count 500 --strict-balance
"""

import argparse
import json
import os
import re
import sys
from collections import Counter
from typing import Dict, List, Tuple

from app.training.dataset import DIFFICULTIES, TOPICS, split_raw_dataset

RAW_PATH = "app/training/data/raw_annotations.json"
TRAIN_PATH = "app/training/data/feedback_dataset_train.json"
EVAL_PATH = "app/training/data/feedback_dataset_eval.json"

REQUIRED_KEYS = {
    "id", "topic", "difficulty", "question", "method",
    "assigned_marks", "total_marks", "steps", "marking_scheme", "step_feedback",
}
STEP_VALIDITIES = {"correct", "partial", "incorrect"}
MARK_EPS = 1e-6

VALIDITY_TARGET = {"correct": 0.30, "partial": 0.35, "incorrect": 0.35}
DIFFICULTY_TARGET = {"easy": 0.30, "medium": 0.45, "difficult": 0.25}
BALANCE_TOLERANCE = 0.08  # ±8 percentage points


def _normalize_question(q: str) -> str:
    """Collapse a question to a template signature: lowercase, digits→#,
    whitespace collapsed. Math operators (=, +, -, *, /, ^, <, >, parens,
    fractions) are DELIBERATELY preserved — stripping them would conflate
    structurally different problems (a linear equation vs an inequality,
    addition vs subtraction) just because they're the same length."""
    q = q.lower()
    q = re.sub(r"\d+(\.\d+)?", "#", q)
    q = re.sub(r"\s+", " ", q).strip()
    return q


def _template_signature(item: Dict) -> Tuple[str, str, Tuple[str, ...]]:
    """A stricter signature than the question template alone: same
    normalized question AND same method AND same per-step validity
    sequence. Two examples sharing this signature are the literal 'same
    template, only the numbers changed, same mistake' case the project
    wants rejected. Sharing just the question template (different errors,
    different mistake location) is legitimate topic-appropriate variety."""
    return (
        _normalize_question(item["question"]),
        item["method"],
        tuple(s["validity"] for s in item["steps"]),
    )


def overall_validity(item: Dict) -> str:
    """Derive an example-level validity label from its step validities —
    there is no stored per-example validity field, by design (see MEMORY /
    conversation: schema keeps validity at the step level)."""
    step_validities = [s["validity"] for s in item["steps"]]
    if all(v == "correct" for v in step_validities):
        return "correct"
    if any(v == "incorrect" for v in step_validities):
        return "incorrect"
    return "partial"


def load_json(path: str) -> List[Dict]:
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def check_schema(raw: List[Dict]) -> List[str]:
    errors = []
    for i, item in enumerate(raw):
        where = f"raw[{i}] (id={item.get('id', '?')})"
        missing = REQUIRED_KEYS - set(item.keys())
        if missing:
            errors.append(f"{where}: missing required keys {sorted(missing)}")
            continue

        if not item["question"].strip():
            errors.append(f"{where}: empty question")
        if not item["method"].strip():
            errors.append(f"{where}: empty method")
        if item["topic"] not in TOPICS:
            errors.append(f"{where}: invalid topic {item['topic']!r}")
        if item["difficulty"] not in DIFFICULTIES:
            errors.append(f"{where}: invalid difficulty {item['difficulty']!r}")
        if not item["steps"]:
            errors.append(f"{where}: empty steps")
        if not item["marking_scheme"]:
            errors.append(f"{where}: empty marking_scheme")
        if not item["step_feedback"]:
            errors.append(f"{where}: empty step_feedback")

        for s in item["steps"]:
            if s["validity"] not in STEP_VALIDITIES:
                errors.append(f"{where}: step {s.get('step_number')} invalid validity {s['validity']!r}")

        # assigned_marks <= total_marks
        if item["assigned_marks"] > item["total_marks"] + MARK_EPS:
            errors.append(
                f"{where}: assigned_marks ({item['assigned_marks']}) > total_marks ({item['total_marks']})"
            )

        # marks_awarded across steps must sum to assigned_marks
        step_sum = sum(s["marks_awarded"] for s in item["steps"])
        if abs(step_sum - item["assigned_marks"]) > MARK_EPS:
            errors.append(
                f"{where}: sum(steps.marks_awarded)={step_sum} != assigned_marks={item['assigned_marks']}"
            )

        # marking_scheme marks must sum to total_marks
        scheme_sum = sum(m["marks"] for m in item["marking_scheme"])
        if abs(scheme_sum - item["total_marks"]) > MARK_EPS:
            errors.append(
                f"{where}: sum(marking_scheme.marks)={scheme_sum} != total_marks={item['total_marks']}"
            )

        # partial/incorrect steps should carry an error_description or missing/deduction feedback
        fb_by_step = {fb["step_number"]: fb for fb in item["step_feedback"]}
        for s in item["steps"]:
            if s["validity"] in ("partial", "incorrect"):
                fb = fb_by_step.get(s["step_number"], {})
                if not s.get("error_description") and not fb.get("missing"):
                    errors.append(
                        f"{where}: step {s['step_number']} is {s['validity']} but has no "
                        f"error_description or feedback.missing explaining why"
                    )
    return errors


def check_duplicates(raw: List[Dict]) -> Tuple[List[str], List[str]]:
    errors = []
    warnings = []

    ids = [item["id"] for item in raw]
    dup_ids = {i for i, c in Counter(ids).items() if c > 1}
    if dup_ids:
        errors.append(f"duplicate ids: {sorted(dup_ids)}")

    questions = [item["question"] for item in raw]
    dup_q = {q for q, c in Counter(questions).items() if c > 1}
    if dup_q:
        errors.append(f"exact duplicate questions: {sorted(dup_q)}")

    # Strict near-duplicate: same normalized question template AND same
    # method AND same per-step validity pattern — i.e. genuinely the same
    # worked example with only the numbers swapped. This is a hard fail.
    sig_counts = Counter(_template_signature(item) for item in raw)
    strict_dupes = {sig: c for sig, c in sig_counts.items() if c > 1}
    if strict_dupes:
        errors.append(
            "near-duplicate examples (same question template, method, and error "
            f"pattern — only numbers differ): {strict_dupes}"
        )

    # Informational only: how often a bare question *shape* repeats, even
    # when the mistake/steps differ (this is expected — many topics need
    # several worked examples that look similar on the surface).
    normalized = Counter(_normalize_question(item["question"]) for item in raw)
    frequent_shapes = {t: c for t, c in normalized.items() if c > 4}
    if frequent_shapes:
        warnings.append(f"question shapes appearing >4x (legitimate if errors/marks vary): {frequent_shapes}")

    return errors, warnings


def check_split_leakage(raw: List[Dict], shipped_train_count: int, shipped_eval_count: int) -> List[str]:
    """Recompute the topic-grouped split from raw (the formatted train/eval
    files only carry {"text": ...} — no id/question — so leakage must be
    checked at the raw level, using the same split function that produced
    the shipped files)."""
    errors = []
    if not raw:
        return errors

    train_raw, eval_raw = split_raw_dataset(raw)

    train_ids = {t["id"] for t in train_raw}
    eval_ids = {t["id"] for t in eval_raw}
    overlap = train_ids & eval_ids
    if overlap:
        errors.append(f"train/eval id overlap: {sorted(overlap)}")

    train_sigs = {_template_signature(t) for t in train_raw}
    eval_sigs = {_template_signature(t) for t in eval_raw}
    sig_overlap = train_sigs & eval_sigs
    if sig_overlap:
        errors.append(
            f"train/eval share {len(sig_overlap)} near-identical example(s) "
            f"(same template, method, and error pattern): {sorted(sig_overlap)[:5]}..."
        )

    if shipped_train_count and len(train_raw) != shipped_train_count:
        errors.append(
            f"shipped feedback_dataset_train.json has {shipped_train_count} examples but "
            f"re-splitting raw_annotations.json now yields {len(train_raw)} — regenerate train/eval"
        )
    if shipped_eval_count and len(eval_raw) != shipped_eval_count:
        errors.append(
            f"shipped feedback_dataset_eval.json has {shipped_eval_count} examples but "
            f"re-splitting raw_annotations.json now yields {len(eval_raw)} — regenerate train/eval"
        )
    return errors


def report_distribution(raw: List[Dict], strict: bool) -> List[str]:
    errors = []
    n = len(raw)
    if n == 0:
        return errors

    print("\n--- Topic coverage ---")
    topic_counts = Counter(item["topic"] for item in raw)
    for t in TOPICS:
        c = topic_counts.get(t, 0)
        flag = "  (NO COVERAGE)" if c == 0 else ""
        print(f"  {t:35s} {c:3d}{flag}")

    print("\n--- Difficulty distribution ---")
    diff_counts = Counter(item["difficulty"] for item in raw)
    for d in DIFFICULTIES:
        pct = diff_counts.get(d, 0) / n
        target = DIFFICULTY_TARGET[d]
        within = abs(pct - target) <= BALANCE_TOLERANCE
        print(f"  {d:10s} {diff_counts.get(d, 0):3d} ({pct:.1%}, target {target:.0%}) {'OK' if within else 'OFF-TARGET'}")
        if strict and not within:
            errors.append(f"difficulty '{d}' at {pct:.1%}, outside target {target:.0%}±{BALANCE_TOLERANCE:.0%}")

    print("\n--- Validity distribution (derived from step validities) ---")
    validity_counts = Counter(overall_validity(item) for item in raw)
    for v in ("correct", "partial", "incorrect"):
        pct = validity_counts.get(v, 0) / n
        target = VALIDITY_TARGET[v]
        within = abs(pct - target) <= BALANCE_TOLERANCE
        print(f"  {v:10s} {validity_counts.get(v, 0):3d} ({pct:.1%}, target {target:.0%}) {'OK' if within else 'OFF-TARGET'}")
        if strict and not within:
            errors.append(f"validity '{v}' at {pct:.1%}, outside target {target:.0%}±{BALANCE_TOLERANCE:.0%}")

    return errors


def main():
    # Windows consoles often default to a non-UTF-8 codepage (e.g. cp1252),
    # which can't encode math symbols (√, etc.) that show up in generated
    # question text — reconfigure so validation output never crashes on them.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser()
    parser.add_argument("--expected-count", type=int, default=None)
    parser.add_argument("--strict-balance", action="store_true")
    args = parser.parse_args()

    raw = load_json(RAW_PATH)
    train = load_json(TRAIN_PATH)
    eval_ = load_json(EVAL_PATH)

    print(f"Loaded {len(raw)} raw examples, {len(train)} train, {len(eval_)} eval")

    errors: List[str] = []
    warnings: List[str] = []

    if args.expected_count is not None and len(raw) != args.expected_count:
        errors.append(f"expected exactly {args.expected_count} raw examples, found {len(raw)}")

    errors += check_schema(raw)
    dup_errors, dup_warnings = check_duplicates(raw)
    errors += dup_errors
    warnings += dup_warnings
    errors += check_split_leakage(raw, len(train), len(eval_))

    dist_errors = report_distribution(raw, strict=args.strict_balance)
    errors += dist_errors

    if warnings:
        print("\n--- Warnings (non-fatal) ---")
        for w in warnings:
            print(f"  WARN: {w}")

    if errors:
        print(f"\n--- FAILED: {len(errors)} error(s) ---")
        for e in errors:
            print(f"  ERROR: {e}")
        sys.exit(1)

    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
