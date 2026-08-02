"""
Base Qwen2.5-3B-Instruct vs the LoRA fine-tune, on the same held-out split.

evaluate.py already scores one configuration at a time (ADAPTER_PATH=none
gives the untuned baseline), but running it twice means loading a 3B model
twice — several minutes each on a free Colab T4 — and then diffing two
summary files by hand. This does both passes in one load and prints the
delta table that goes in the report.

The adapter is attached once and toggled per example with PeftModel's
disable_adapter() context, so the two passes share identical weights,
tokenizer, prompt and decoding settings. The only difference between them
is whether the LoRA deltas are active — which is exactly the claim the
comparison is supposed to support.

Both passes decode greedily (do_sample=False) with the same token budget,
so the numbers are reproducible and like-for-like. Note that the base model
tends to ramble past the format, and a truncated completion scores as badly
as a malformed one; that is a real property of the untuned model on this
prompt, not an artefact — raise MAX_NEW_TOKENS for both sides if you want
to show it is not purely a length effect.

Usage (Colab, after app.training.train):
    python -m tests.evaluation_suite.compare

Environment:
    BASE_MODEL         default Qwen/Qwen2.5-3B-Instruct
    ADAPTER_PATH       local adapter dir or Hub repo id (default ./lora-adapter)
    EVAL_DATASET_PATH  default app/training/data/feedback_dataset_eval.json
    RESULTS_DIR        default tests/evaluation_suite/results
    MAX_NEW_TOKENS     default 300, applied to both passes
    HF_TOKEN           only needed when ADAPTER_PATH is a private Hub repo
    LIMIT              compare only the first N examples (smoke test)

Writes, under RESULTS_DIR:
    base/summary-<stamp>.json       ) same shape evaluate.py writes, so
    base/predictions-<stamp>.json   ) plots.py reads them unchanged:
    tuned/summary-<stamp>.json      )   python -m tests.evaluation_suite.plots \
    tuned/predictions-<stamp>.json  )     --results .../tuned --baseline .../base
    comparison-<stamp>.json         the delta table + paired per-example wins
"""

import json
import os
import time
from typing import Dict, List, Optional, Tuple

from tests.evaluation_suite.evaluate import (
    VALIDITIES,
    aggregate,
    generate,
    load_examples,
    score_example,
)

# Metrics reported side by side. Every one is "higher is better" except the
# violation rate, hence the flag — the delta column needs to know which way
# an improvement points.
_COMPARABLE: List[Tuple[str, str, bool]] = [
    ("format_valid_rate", "Format compliance", True),
    ("step_count_match_rate", "Step count match", True),
    ("validity_accuracy", "Validity accuracy", True),
    ("validity_macro_f1", "Validity macro F1", True),
    ("field_rule_violation_rate", "Field-rule violations", False),
]


# --------------------------------------------------------------------------
# Paired per-example comparison
# --------------------------------------------------------------------------


def example_accuracy(scored: Dict) -> Optional[float]:
    """Validity accuracy for one example, over *reference* steps.

    Deliberately stricter than the aggregate `validity_accuracy`, which
    divides by the steps that aligned: here a reference step the model never
    produced counts as wrong. Otherwise a completion with no parseable steps
    at all — the base model's usual failure on this prompt — would have no
    comparable value and drop out of the win/loss counts entirely, scoring
    "not applicable" rather than the loss it plainly is.

    None only when the reference itself has no steps to score against.
    """
    pairs = scored["validity_pairs"]
    if not pairs:
        return None
    return sum(1 for ref, pred in pairs if ref == pred) / len(pairs)


def example_rouge(scored: Dict) -> Optional[float]:
    """Mean ROUGE-L across this example's reference fields.

    Zero — not None — when the reference has steps but the prediction gave
    nothing to match against them, for the same reason as above.
    """
    if not scored["ref_step_count"]:
        return None
    values = list(scored["rouge"].values())
    return sum(values) / len(values) if values else 0.0


def _win_loss(base_values: List[Optional[float]], tuned_values: List[Optional[float]]) -> Dict:
    """Count examples where the fine-tune beats / ties / loses to the base.

    Examples where either side has no comparable value (nothing aligned, no
    reference fields) are skipped rather than counted as ties — a tie should
    mean the two models genuinely scored the same.
    """
    wins = ties = losses = 0
    for base_value, tuned_value in zip(base_values, tuned_values):
        if base_value is None or tuned_value is None:
            continue
        if tuned_value > base_value:
            wins += 1
        elif tuned_value < base_value:
            losses += 1
        else:
            ties += 1
    return {"tuned_better": wins, "tie": ties, "base_better": losses,
            "compared": wins + ties + losses}


def build_comparison(base: Dict, tuned: Dict, base_results: List[Dict],
                     tuned_results: List[Dict]) -> Dict:
    """Delta table + paired win/loss counts for the two summaries."""
    metrics = {}
    for key, label, higher_is_better in _COMPARABLE:
        base_value, tuned_value = base[key], tuned[key]
        delta = tuned_value - base_value
        metrics[key] = {
            "label": label,
            "base": base_value,
            "tuned": tuned_value,
            "delta": delta,
            "improved": delta > 0 if higher_is_better else delta < 0,
            "higher_is_better": higher_is_better,
        }

    # ROUGE-L is per field, and a field only appears if some reference used
    # it — union the two sides so a field only one model produced still shows.
    rouge_fields = sorted(set(base["rouge_l"]) | set(tuned["rouge_l"]))
    rouge = {}
    for name in rouge_fields:
        base_value = base["rouge_l"].get(name, 0.0)
        tuned_value = tuned["rouge_l"].get(name, 0.0)
        rouge[name] = {"base": base_value, "tuned": tuned_value,
                       "delta": tuned_value - base_value}

    per_class = {}
    for label in VALIDITIES:
        base_f1 = base["validity_per_class"][label]["f1"]
        tuned_f1 = tuned["validity_per_class"][label]["f1"]
        per_class[label] = {"base_f1": base_f1, "tuned_f1": tuned_f1,
                            "delta": tuned_f1 - base_f1,
                            "support": tuned["validity_per_class"][label]["support"]}

    return {
        "examples": tuned["examples"],
        "metrics": metrics,
        "rouge_l": rouge,
        "validity_per_class_f1": per_class,
        "paired": {
            "validity_accuracy": _win_loss(
                [example_accuracy(r) for r in base_results],
                [example_accuracy(r) for r in tuned_results],
            ),
            "rouge_l": _win_loss(
                [example_rouge(r) for r in base_results],
                [example_rouge(r) for r in tuned_results],
            ),
        },
    }


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------


def load_model(base_model: str, adapter_path: str):
    """Load the base model with the adapter attached but toggleable."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("no CUDA device — run this on a Colab/Kaggle GPU runtime")
    if adapter_path.lower() in ("none", ""):
        raise ValueError(
            "ADAPTER_PATH=none has nothing to compare against the base model — "
            "point it at ./lora-adapter or your Hub repo id"
        )

    # T4 (Turing) has no native bfloat16; fp16 is the correct choice there.
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=dtype)
    model = PeftModel.from_pretrained(model, adapter_path, token=os.getenv("HF_TOKEN"))
    model = model.to("cuda")
    model.eval()
    print(f"Loaded {base_model} + adapter {adapter_path} ({str(dtype).split('.')[-1]})")
    return model, tokenizer


# --------------------------------------------------------------------------
# Runner
# --------------------------------------------------------------------------


def _write(results_dir: str, name: str, stamp: str, payload) -> str:
    os.makedirs(results_dir, exist_ok=True)
    path = os.path.join(results_dir, f"{name}-{stamp}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return path


def main() -> None:
    base_model = os.getenv("BASE_MODEL", "Qwen/Qwen2.5-3B-Instruct")
    adapter_path = os.getenv("ADAPTER_PATH", "./lora-adapter")
    eval_path = os.getenv("EVAL_DATASET_PATH", "app/training/data/feedback_dataset_eval.json")
    results_dir = os.getenv("RESULTS_DIR", "tests/evaluation_suite/results")
    max_new_tokens = int(os.getenv("MAX_NEW_TOKENS", "300"))
    limit = int(os.getenv("LIMIT", "0")) or None

    examples = load_examples(eval_path, limit)
    print(f"Comparing base vs fine-tuned on {len(examples)} examples from {eval_path}\n")

    model, tokenizer = load_model(base_model, adapter_path)

    base_results, tuned_results = [], []
    base_records, tuned_records = [], []
    base_seconds = tuned_seconds = 0.0
    started = time.time()

    # Interleaved rather than two full passes: if the Colab runtime drops
    # mid-run, what has been generated so far is still a matched pair.
    for i, (prompt, reference) in enumerate(examples, 1):
        mark = time.time()
        with model.disable_adapter():
            base_prediction = generate(model, tokenizer, prompt, max_new_tokens)
        base_seconds += time.time() - mark

        mark = time.time()
        tuned_prediction = generate(model, tokenizer, prompt, max_new_tokens)
        tuned_seconds += time.time() - mark

        for prediction, results, records in (
            (base_prediction, base_results, base_records),
            (tuned_prediction, tuned_results, tuned_records),
        ):
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
            print(f"  {i}/{len(examples)}  ({elapsed / i:.1f}s per example, both models)")

    base_summary = aggregate(base_results)
    base_summary.update({"base_model": base_model, "adapter_path": "none",
                         "seconds": round(base_seconds, 1),
                         "max_new_tokens": max_new_tokens})
    tuned_summary = aggregate(tuned_results)
    tuned_summary.update({"base_model": base_model, "adapter_path": adapter_path,
                          "seconds": round(tuned_seconds, 1),
                          "max_new_tokens": max_new_tokens})

    comparison = build_comparison(base_summary, tuned_summary, base_results, tuned_results)
    comparison.update({
        "base_model": base_model,
        "adapter_path": adapter_path,
        "eval_dataset": eval_path,
        "max_new_tokens": max_new_tokens,
        "seconds": {"base": round(base_seconds, 1), "tuned": round(tuned_seconds, 1)},
    })

    stamp = time.strftime("%Y%m%d-%H%M%S")
    base_dir = os.path.join(results_dir, "base")
    tuned_dir = os.path.join(results_dir, "tuned")
    written = [
        _write(base_dir, "summary", stamp, base_summary),
        _write(base_dir, "predictions", stamp, base_records),
        _write(tuned_dir, "summary", stamp, tuned_summary),
        _write(tuned_dir, "predictions", stamp, tuned_records),
        _write(results_dir, "comparison", stamp, comparison),
    ]

    print_comparison(comparison)
    print()
    for path in written:
        print(f"wrote {path}")
    print(
        "\nFigures:\n"
        f"  python -m tests.evaluation_suite.plots --results {tuned_dir} --baseline {base_dir}"
    )


def print_comparison(c: Dict) -> None:
    seconds = c["seconds"]
    print("\n" + "=" * 66)
    print(f"BASE vs FINE-TUNED — {c['examples']} held-out examples")
    print(f"{c['base_model']}  vs  + {c['adapter_path']}")
    print("=" * 66)

    print(f"\n{'metric':<26}{'base':>10}{'tuned':>10}{'delta':>12}")
    print("-" * 58)
    for entry in c["metrics"].values():
        arrow = "improved" if entry["improved"] else ("same" if entry["delta"] == 0 else "worse")
        print(f"  {entry['label']:<24}{entry['base']:>10.3f}{entry['tuned']:>10.3f}"
              f"{entry['delta']:>+9.3f}  {arrow}")

    print(f"\n{'validity F1 by class':<26}{'base':>10}{'tuned':>10}{'delta':>12}{'support':>10}")
    print("-" * 68)
    for label, entry in c["validity_per_class_f1"].items():
        print(f"  {label:<24}{entry['base_f1']:>10.3f}{entry['tuned_f1']:>10.3f}"
              f"{entry['delta']:>+9.3f}   {entry['support']:>9}")

    if c["rouge_l"]:
        print(f"\n{'ROUGE-L by field':<26}{'base':>10}{'tuned':>10}{'delta':>12}")
        print("-" * 58)
        for name, entry in c["rouge_l"].items():
            print(f"  {name:<24}{entry['base']:>10.3f}{entry['tuned']:>10.3f}"
                  f"{entry['delta']:>+9.3f}")

    print("\n--- Per-example wins (fine-tuned vs base) ---")
    for name, counts in c["paired"].items():
        print(f"  {name:<20} tuned {counts['tuned_better']:>3}  "
              f"tie {counts['tie']:>3}  base {counts['base_better']:>3}  "
              f"(of {counts['compared']} comparable)")

    print("\n--- Generation cost ---")
    print(f"  base    {seconds['base']:>8.1f}s total")
    print(f"  tuned   {seconds['tuned']:>8.1f}s total")


if __name__ == "__main__":
    main()
