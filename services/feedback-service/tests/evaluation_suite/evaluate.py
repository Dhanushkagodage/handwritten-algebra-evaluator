"""
Offline evaluation of the fine-tuned feedback model against the held-out
eval split.

Runs the model directly (base + LoRA adapter) instead of calling the
Hugging Face Space, so it costs nothing on a free Colab/Kaggle T4 and does
not consume the account's ZeroGPU quota. Decoding is greedy — the Space
samples at temperature 0.7 for variety, but eval needs the run to be
reproducible.

Usage (Colab, straight after app.training.train):
    python -m tests.evaluation_suite.evaluate

Environment:
    ADAPTER_PATH  local adapter dir or Hub repo id (default ./lora-adapter)
    BASE_MODEL    default Qwen/Qwen2.5-3B-Instruct
    EVAL_DATASET_PATH  default app/training/data/feedback_dataset_eval.json
    RESULTS_DIR   default tests/evaluation_suite/results
    HF_TOKEN      only needed when ADAPTER_PATH is a private Hub repo
    LIMIT         evaluate only the first N examples (smoke test)
"""

import json
import os
import re
import time
from collections import Counter
from typing import Dict, List, Optional, Tuple

# torch/peft/transformers are imported lazily inside load_model() and
# generate() so the parsing and metric functions stay importable (and
# testable) on a machine with no torch installed.

# Must match app/training/dataset.py — the eval JSON stores prompt and
# reference concatenated in one "text" field, split on the assistant turn.
_ASSISTANT_MARKER = "<|im_start|>assistant\n"
_IM_END = "<|im_end|>"

_STEP_HEADER = re.compile(r"^===\s*STEP\s+(\d+)\s*\[(CORRECT|PARTIAL|INCORRECT)\]\s*===$")
_FIELD = re.compile(r"^(CORRECT|MISSING|DEDUCTION|IMPROVE):\s*(.*)$")

VALIDITIES = ("CORRECT", "PARTIAL", "INCORRECT")
# Fields the prompt asks for on every step vs only on non-CORRECT steps.
_ALWAYS_FIELDS = ("CORRECT", "IMPROVE")
_ERROR_ONLY_FIELDS = ("MISSING", "DEDUCTION")


# --------------------------------------------------------------------------
# Parsing
# --------------------------------------------------------------------------


def parse_feedback(text: str) -> Optional[List[Dict]]:
    """Parse a === STEP N [VALIDITY] === completion into step dicts.

    Returns None when the text contains no recognisable step header at all,
    which is the signal that the model ignored the output format.
    """
    steps: List[Dict] = []
    current: Optional[Dict] = None
    last_field: Optional[str] = None

    for raw_line in text.splitlines():
        line = raw_line.strip()

        header = _STEP_HEADER.match(line)
        if header:
            if current is not None:
                steps.append(current)
            current = {
                "step_number": int(header.group(1)),
                "validity": header.group(2),
                "fields": {},
            }
            last_field = None
            continue

        if current is None:
            continue

        field = _FIELD.match(line)
        if field:
            name, value = field.group(1), field.group(2).strip()
            current["fields"][name] = value
            last_field = name
        elif line and last_field:
            # Continuation of a wrapped field value.
            current["fields"][last_field] = f"{current['fields'][last_field]} {line}".strip()

    if current is not None:
        steps.append(current)

    return steps or None


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------


def _tokens(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def rouge_l(prediction: str, reference: str) -> float:
    """ROUGE-L F1 over word tokens.

    Implemented here rather than pulling in rouge-score: requirements-train.txt
    is version-pinned against Colab's preinstalled stack and adding
    dependencies to it risks breaking training.
    """
    pred, ref = _tokens(prediction), _tokens(reference)
    if not pred or not ref:
        return 0.0

    # Rolling two-row LCS — full table is unnecessary and 80x slower here.
    previous = [0] * (len(ref) + 1)
    for p in pred:
        current = [0]
        for j, r in enumerate(ref):
            current.append(previous[j] + 1 if p == r else max(current[j], previous[j + 1]))
        previous = current

    lcs = previous[-1]
    if lcs == 0:
        return 0.0
    precision, recall = lcs / len(pred), lcs / len(ref)
    return 2 * precision * recall / (precision + recall)


def _prf(true_positive: int, false_positive: int, false_negative: int) -> Dict[str, float]:
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def score_example(prediction: str, reference: str) -> Dict:
    """Compare one generated completion against its reference."""
    pred_steps = parse_feedback(prediction)
    ref_steps = parse_feedback(reference) or []

    result = {
        "parsed": pred_steps is not None,
        "ref_step_count": len(ref_steps),
        "pred_step_count": len(pred_steps) if pred_steps else 0,
        "step_count_match": False,
        "validity_pairs": [],       # (reference, predicted) for aligned steps
        "field_rule_violations": [],
        "rouge": {},
    }
    if pred_steps is None:
        # Still record the reference steps as unmatched so an unparseable
        # completion shows up in the step counts rather than silently
        # shrinking the denominator.
        result["validity_pairs"] = [(s["validity"], None) for s in ref_steps]
        return result

    result["step_count_match"] = len(pred_steps) == len(ref_steps)

    # Align on step_number where possible; the model occasionally renumbers,
    # so fall back to positional alignment for the remainder.
    pred_by_number = {s["step_number"]: s for s in pred_steps}
    rouge_totals: Dict[str, List[float]] = {}

    for index, ref_step in enumerate(ref_steps):
        pred_step = pred_by_number.get(ref_step["step_number"])
        if pred_step is None:
            pred_step = pred_steps[index] if index < len(pred_steps) else None
        if pred_step is None:
            result["validity_pairs"].append((ref_step["validity"], None))
            continue

        result["validity_pairs"].append((ref_step["validity"], pred_step["validity"]))

        # The prompt requires CORRECT/IMPROVE always, and MISSING/DEDUCTION
        # only when the step is not fully correct.
        for name in _ALWAYS_FIELDS:
            if not pred_step["fields"].get(name):
                result["field_rule_violations"].append(f"step {ref_step['step_number']}: missing {name}")
        for name in _ERROR_ONLY_FIELDS:
            present = bool(pred_step["fields"].get(name))
            expected = pred_step["validity"] != "CORRECT"
            if expected and not present:
                result["field_rule_violations"].append(f"step {ref_step['step_number']}: missing {name}")
            elif not expected and present:
                result["field_rule_violations"].append(f"step {ref_step['step_number']}: unexpected {name}")

        for name in _ALWAYS_FIELDS + _ERROR_ONLY_FIELDS:
            ref_value = ref_step["fields"].get(name)
            if not ref_value:
                continue
            rouge_totals.setdefault(name, []).append(
                rouge_l(pred_step["fields"].get(name, ""), ref_value)
            )

    result["rouge"] = {k: sum(v) / len(v) for k, v in rouge_totals.items() if v}
    return result


def aggregate(results: List[Dict]) -> Dict:
    """Roll per-example results up into the reported summary."""
    total = len(results)
    parsed = [r for r in results if r["parsed"]]

    pairs = [p for r in results for p in r["validity_pairs"]]
    matched = [(ref, pred) for ref, pred in pairs if pred is not None]
    correct_labels = sum(1 for ref, pred in matched if ref == pred)

    per_class = {}
    for label in VALIDITIES:
        true_positive = sum(1 for ref, pred in matched if ref == label and pred == label)
        false_positive = sum(1 for ref, pred in matched if ref != label and pred == label)
        false_negative = sum(1 for ref, pred in matched if ref == label and pred != label)
        per_class[label] = {**_prf(true_positive, false_positive, false_negative),
                            "support": sum(1 for ref, _ in matched if ref == label)}

    confusion = Counter((ref, pred) for ref, pred in matched)

    rouge_fields: Dict[str, List[float]] = {}
    for r in results:
        for name, value in r["rouge"].items():
            rouge_fields.setdefault(name, []).append(value)

    return {
        "examples": total,
        "format_valid_rate": len(parsed) / total if total else 0.0,
        "step_count_match_rate": sum(r["step_count_match"] for r in results) / total if total else 0.0,
        "steps_evaluated": len(matched),
        "steps_unmatched": len(pairs) - len(matched),
        "validity_accuracy": correct_labels / len(matched) if matched else 0.0,
        "validity_macro_f1": sum(per_class[l]["f1"] for l in VALIDITIES) / len(VALIDITIES),
        "validity_per_class": per_class,
        "confusion": {f"{ref}->{pred}": n for (ref, pred), n in sorted(confusion.items())},
        "field_rule_violation_rate": sum(bool(r["field_rule_violations"]) for r in results) / total if total else 0.0,
        "rouge_l": {k: sum(v) / len(v) for k, v in rouge_fields.items()},
    }


# --------------------------------------------------------------------------
# Runner
# --------------------------------------------------------------------------


def load_examples(path: str, limit: Optional[int] = None) -> List[Tuple[str, str]]:
    """Split each eval record's `text` into (prompt, reference completion)."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if limit:
        data = data[:limit]

    examples = []
    for i, item in enumerate(data):
        text = item["text"]
        if _ASSISTANT_MARKER not in text:
            raise ValueError(f"example {i} has no assistant turn — regenerate the dataset")
        prompt, reference = text.split(_ASSISTANT_MARKER, 1)
        examples.append((prompt + _ASSISTANT_MARKER, reference.replace(_IM_END, "").strip()))
    return examples


def load_model(base_model: str, adapter_path: str):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("no CUDA device — run this on a Colab/Kaggle GPU runtime")

    # T4 (Turing) has no native bfloat16; fp16 is the correct choice there.
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=dtype)
    # ADAPTER_PATH=none evaluates the untuned base model, which is the
    # baseline the fine-tuned numbers should be reported against.
    if adapter_path.lower() in ("none", ""):
        label = "no adapter (base-model baseline)"
    else:
        model = PeftModel.from_pretrained(model, adapter_path, token=os.getenv("HF_TOKEN"))
        label = f"adapter {adapter_path}"
    model = model.to("cuda")
    model.eval()
    print(f"Loaded {base_model} + {label} ({str(dtype).split('.')[-1]})")
    return model, tokenizer


def generate(model, tokenizer, prompt: str, max_new_tokens: int = 300) -> str:
    import torch

    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # greedy — eval must be reproducible
            pad_token_id=tokenizer.eos_token_id,
        )
    text = tokenizer.decode(
        output_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
    )
    return text.strip()


def main() -> None:
    base_model = os.getenv("BASE_MODEL", "Qwen/Qwen2.5-3B-Instruct")
    adapter_path = os.getenv("ADAPTER_PATH", "./lora-adapter")
    eval_path = os.getenv("EVAL_DATASET_PATH", "app/training/data/feedback_dataset_eval.json")
    results_dir = os.getenv("RESULTS_DIR", "tests/evaluation_suite/results")
    limit = int(os.getenv("LIMIT", "0")) or None

    examples = load_examples(eval_path, limit)
    print(f"Evaluating {len(examples)} examples from {eval_path}\n")

    model, tokenizer = load_model(base_model, adapter_path)

    results, records = [], []
    started = time.time()
    for i, (prompt, reference) in enumerate(examples, 1):
        prediction = generate(model, tokenizer, prompt)
        scored = score_example(prediction, reference)
        results.append(scored)
        records.append({
            "index": i - 1,
            "prompt": prompt,
            "reference": reference,
            "prediction": prediction,
            "scores": scored,
        })
        if i % 10 == 0 or i == len(examples):
            elapsed = time.time() - started
            print(f"  {i}/{len(examples)}  ({elapsed / i:.1f}s per example)")

    summary = aggregate(results)
    summary["base_model"] = base_model
    summary["adapter_path"] = adapter_path
    summary["seconds"] = round(time.time() - started, 1)

    os.makedirs(results_dir, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    summary_path = os.path.join(results_dir, f"summary-{stamp}.json")
    detail_path = os.path.join(results_dir, f"predictions-{stamp}.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    with open(detail_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)

    print_summary(summary)
    print(f"\nSummary     → {summary_path}")
    print(f"Predictions → {detail_path}")


def print_summary(s: Dict) -> None:
    print("\n" + "=" * 58)
    print(f"EVALUATION — {s['examples']} examples, {s['steps_evaluated']} steps, {s['seconds']}s")
    print("=" * 58)

    print("\n--- Format compliance ---")
    print(f"  parses into === STEP N ===      {s['format_valid_rate']:.1%}")
    print(f"  step count matches reference    {s['step_count_match_rate']:.1%}")
    print(f"  examples with field-rule issues {s['field_rule_violation_rate']:.1%}")
    if s["steps_unmatched"]:
        print(f"  reference steps with no prediction: {s['steps_unmatched']}")

    print("\n--- Validity classification ---")
    print(f"  accuracy   {s['validity_accuracy']:.1%}")
    print(f"  macro F1   {s['validity_macro_f1']:.3f}")
    print(f"  {'label':<12}{'prec':>8}{'recall':>8}{'f1':>8}{'support':>9}")
    for label in VALIDITIES:
        m = s["validity_per_class"][label]
        print(f"  {label:<12}{m['precision']:>8.3f}{m['recall']:>8.3f}{m['f1']:>8.3f}{m['support']:>9}")

    print("\n--- Confusion (reference -> predicted) ---")
    for key, count in s["confusion"].items():
        print(f"  {key:<26}{count}")

    print("\n--- ROUGE-L vs reference text ---")
    for name, value in sorted(s["rouge_l"].items()):
        print(f"  {name:<12}{value:.3f}")


if __name__ == "__main__":
    main()
