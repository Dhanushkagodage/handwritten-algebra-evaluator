"""
Render the evaluation metrics from evaluate.py as report figures.

Reads the summary-*.json written by tests.evaluation_suite.evaluate and
writes PNGs alongside it. Pass a baseline summary (the ADAPTER_PATH=none
run) to also get the base-vs-fine-tuned comparison.

Usage:
    python -m tests.evaluation_suite.plots
    python -m tests.evaluation_suite.plots --baseline tests/evaluation_suite/results/baseline

Environment / flags:
    --results DIR    directory holding summary-*.json (default
                     tests/evaluation_suite/results); the newest is used
    --baseline DIR   optional baseline results dir for figure 5
    --dpi N          output resolution (default 200)

Figures are committed to light mode only — these land in a written report
and in print, not on a themed web page.
"""

import argparse
import glob
import json
import os
from typing import Dict, List, Optional, Tuple

# --------------------------------------------------------------------------
# Palette — light mode, from the design system's validated categorical order.
# Slots 1-3 clear every all-pairs gate on the light surface (worst CVD
# deltaE 9.2, worst normal-vision deltaE 24.0). AQUA sits at 2.74:1 against
# the surface, below 3:1, so the relief rule applies: the series it carries
# is always direct-labelled.
# --------------------------------------------------------------------------
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

BLUE = "#2a78d6"      # categorical slot 1
ORANGE = "#eb6834"    # categorical slot 2
AQUA = "#1baf7a"      # categorical slot 3 — needs direct labels

# Single-hue sequential ramp (blue, light to dark) for the confusion heatmap.
BLUE_RAMP = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec",
             "#5598e7", "#3987e5", "#2a78d6", "#256abf", "#1c5cab",
             "#184f95", "#104281", "#0d366b"]
# Two shades of one hue for the before/after dumbbell.
BLUE_LIGHT = "#86b6ef"

VALIDITIES = ("CORRECT", "PARTIAL", "INCORRECT")

# Mark spec: bars never fill their band. Measured in points (72/inch) rather
# than device pixels so the cap means the same thing at any --dpi; 18pt is
# the 24px spec at a nominal 96dpi.
BAR_MAX_PT = 18
BAR_GAP_PT = 2   # the surface gap that separates adjacent bars


def _mpl():
    """Import matplotlib with the non-interactive backend (headless/Colab)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Arial"],
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "text.color": INK,
        "axes.labelcolor": INK_SECONDARY,
        "xtick.color": INK_MUTED,
        "ytick.color": INK_MUTED,
        "axes.edgecolor": AXIS,
        "axes.linewidth": 1.0,
        "xtick.major.size": 0,
        "ytick.major.size": 0,
        "font.size": 10,
    })
    return plt


def _style_axes(ax, value_axis: str = "y", baseline: bool = True) -> None:
    """Hairline, solid, recessive grid on the value axis only; no box.

    `baseline=False` for marks that don't grow from zero (dumbbells), where a
    spine at the axis edge would imply an origin the marks aren't measured from.
    """
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)
    if baseline:
        ax.spines["bottom" if value_axis == "y" else "left"].set_visible(True)
    ax.grid(axis=value_axis, color=GRID, linewidth=1.0, linestyle="-", zorder=0)
    ax.set_axisbelow(True)


def _bar_geometry(extent_in: float, plot_fraction: float, n_bands: int,
                  n_series: int) -> Tuple[float, float]:
    """Bar thickness and per-series offset step, both in band units (band = 1.0).

    Caps thickness at BAR_MAX_PT so bars stay thin, and reserves BAR_GAP_PT of
    surface between neighbours rather than drawing a border around them.
    """
    band_pt = (extent_in * 72 * plot_fraction) / max(n_bands, 1)
    thickness_pt = min(BAR_MAX_PT, (band_pt * 0.82) / n_series)
    return thickness_pt / band_pt, (thickness_pt + BAR_GAP_PT) / band_pt


# --------------------------------------------------------------------------
# Figure 1 — headline numbers as a KPI row, not a bar chart
# --------------------------------------------------------------------------


def fig_headline(summary: Dict, path: str, dpi: int) -> str:
    plt = _mpl()
    tiles = [
        ("Validity accuracy", f"{summary['validity_accuracy']:.1%}",
         f"{summary['steps_evaluated']} steps scored"),
        ("Macro F1", f"{summary['validity_macro_f1']:.3f}",
         "across 3 validity classes"),
        ("Format compliance", f"{summary['format_valid_rate']:.1%}",
         "parses as === STEP N ==="),
        ("Field-rule violations", f"{summary['field_rule_violation_rate']:.1%}",
         "of examples (lower is better)"),
    ]

    fig, axes = plt.subplots(1, len(tiles), figsize=(11, 2.2))
    for ax, (label, value, sub) in zip(axes, tiles):
        ax.axis("off")
        ax.text(0, 0.86, label, fontsize=10, color=INK_SECONDARY, va="top")
        # Proportional figures (never tabular) on a large standalone number.
        ax.text(0, 0.52, value, fontsize=30, color=INK, va="center", fontweight="semibold")
        ax.text(0, 0.10, sub, fontsize=8.5, color=INK_MUTED, va="bottom")

    fig.suptitle(
        f"Feedback model evaluation — {summary['examples']} held-out examples",
        fontsize=13, color=INK, x=0.5, y=1.04, fontweight="semibold",
    )
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path


# --------------------------------------------------------------------------
# Figure 2 — per-class precision / recall / F1 (grouped bar, 3 series)
# --------------------------------------------------------------------------


def fig_per_class(summary: Dict, path: str, dpi: int) -> str:
    plt = _mpl()
    import numpy as np

    per_class = summary["validity_per_class"]
    series = [("Precision", BLUE), ("Recall", ORANGE), ("F1", AQUA)]
    keys = ["precision", "recall", "f1"]

    fig_w = 6.6
    fig, ax = plt.subplots(figsize=(fig_w, 3.9))
    x = np.arange(len(VALIDITIES))
    width, step = _bar_geometry(fig_w, 0.72, len(VALIDITIES), len(series))

    for i, ((name, color), key) in enumerate(zip(series, keys)):
        offset = (i - (len(series) - 1) / 2) * step
        values = [per_class[v][key] for v in VALIDITIES]
        ax.bar(x + offset, values, width, label=name, color=color, zorder=2)
        # Relief for AQUA (2.74:1): the F1 series is always directly labelled.
        # Precision/recall are carried by the axis, the legend and the table.
        if name == "F1":
            for xi, v in zip(x + offset, values):
                ax.text(xi, v + 0.025, f"{v:.2f}", ha="center", va="bottom",
                        fontsize=8.5, color=INK_SECONDARY)

    ax.set_xticks(x)
    ax.set_xticklabels([f"{v}\n{per_class[v]['support']} steps" for v in VALIDITIES],
                       fontsize=9.5, color=INK_SECONDARY)
    ax.set_ylim(0, 1.08)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0", "0.25", "0.50", "0.75", "1.00"], fontsize=9)
    ax.set_ylabel("Score", fontsize=9.5)
    _style_axes(ax, "y")

    legend = ax.legend(frameon=False, ncol=3, loc="upper center",
                       bbox_to_anchor=(0.5, 1.13), fontsize=9.5)
    for text in legend.get_texts():
        text.set_color(INK_SECONDARY)   # text never wears the data color

    ax.set_title("Step validity classification, per class", fontsize=12,
                 color=INK, pad=34, loc="left", fontweight="semibold")
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path


# --------------------------------------------------------------------------
# Figure 3 — confusion matrix (heatmap, one-hue sequential)
# --------------------------------------------------------------------------


def _ramp_color(fraction: float) -> Tuple[str, str]:
    """Pick a step from the blue ramp plus a legible ink for text on it."""
    index = int(round(max(0.0, min(1.0, fraction)) * (len(BLUE_RAMP) - 1)))
    fill = BLUE_RAMP[index]
    r, g, b = (int(fill[i:i + 2], 16) / 255 for i in (1, 3, 5))
    luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return fill, (INK if luminance > 0.55 else "#ffffff")


def fig_confusion(summary: Dict, path: str, dpi: int) -> str:
    plt = _mpl()

    counts = {(a, b): 0 for a in VALIDITIES for b in VALIDITIES}
    for key, n in summary["confusion"].items():
        ref, pred = key.split("->")
        if (ref, pred) in counts:
            counts[(ref, pred)] = n

    fig, ax = plt.subplots(figsize=(5.6, 4.6))
    largest = max(counts.values()) or 1

    for row, ref in enumerate(VALIDITIES):
        row_total = sum(counts[(ref, p)] for p in VALIDITIES) or 1
        for col, pred in enumerate(VALIDITIES):
            n = counts[(ref, pred)]
            fill, text_color = _ramp_color(n / largest)
            # 2px surface gap does the separating — no borders on the cells.
            ax.add_patch(plt.Rectangle((col + 0.015, row + 0.015), 0.97, 0.97,
                                       facecolor=fill, edgecolor="none", zorder=2))
            ax.text(col + 0.5, row + 0.42, str(n), ha="center", va="center",
                    fontsize=15, color=text_color, zorder=3, fontweight="semibold")
            ax.text(col + 0.5, row + 0.72, f"{n / row_total:.0%}", ha="center",
                    va="center", fontsize=8.5, color=text_color, zorder=3)

    ax.set_xlim(0, 3)
    ax.set_ylim(3, 0)
    ax.set_xticks([i + 0.5 for i in range(3)])
    ax.set_yticks([i + 0.5 for i in range(3)])
    ax.set_xticklabels(VALIDITIES, fontsize=9, color=INK_SECONDARY)
    ax.set_yticklabels(VALIDITIES, fontsize=9, color=INK_SECONDARY)
    ax.set_xlabel("Predicted", fontsize=9.5, labelpad=8)
    ax.set_ylabel("Reference", fontsize=9.5, labelpad=8)
    ax.xaxis.set_label_position("top")
    ax.xaxis.tick_top()
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)
    ax.tick_params(length=0)

    ax.set_title("Confusion matrix — cell shade and % are row-normalised",
                 fontsize=11.5, color=INK, pad=30, loc="left", fontweight="semibold")
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path


# --------------------------------------------------------------------------
# Figure 4 — ROUGE-L per field (single series, so no legend)
# --------------------------------------------------------------------------


def fig_rouge(summary: Dict, path: str, dpi: int) -> str:
    plt = _mpl()
    import numpy as np

    rouge = summary.get("rouge_l") or {}
    if not rouge:
        return ""
    fields = sorted(rouge, key=lambda k: rouge[k])
    values = [rouge[f] for f in fields]

    fig_h = 0.42 * len(fields) + 1.5
    fig, ax = plt.subplots(figsize=(6.8, fig_h))
    y = np.arange(len(fields))
    height, _ = _bar_geometry(fig_h, 0.62, len(fields), 1)
    # Nominal categories -> one hue for every bar, never a value ramp.
    ax.barh(y, values, height=height, color=BLUE, zorder=2)

    for yi, v in zip(y, values):
        ax.text(v + 0.015, yi, f"{v:.3f}", va="center", ha="left",
                fontsize=9, color=INK_SECONDARY)

    ax.set_yticks(y)
    ax.set_yticklabels(fields, fontsize=9.5, color=INK_SECONDARY)
    ax.set_xlim(0, min(1.0, max(values) * 1.25))
    ax.set_xlabel("ROUGE-L F1 vs reference text", fontsize=9.5)
    _style_axes(ax, "x")

    ax.set_title("Wording overlap with the reference feedback, by field",
                 fontsize=12, color=INK, pad=14, loc="left", fontweight="semibold")
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path


# --------------------------------------------------------------------------
# Figure 5 — base model vs fine-tuned (dumbbell: before -> after per item)
# --------------------------------------------------------------------------


def _comparable(summary: Dict) -> List[Tuple[str, float]]:
    rouge = summary.get("rouge_l") or {}
    rows = [
        ("Format compliance", summary["format_valid_rate"]),
        ("Step count match", summary["step_count_match_rate"]),
        ("Validity accuracy", summary["validity_accuracy"]),
        ("Macro F1", summary["validity_macro_f1"]),
    ]
    if rouge:
        rows.append(("ROUGE-L (mean)", sum(rouge.values()) / len(rouge)))
    return rows


def fig_baseline(summary: Dict, baseline: Dict, path: str, dpi: int) -> str:
    plt = _mpl()
    import numpy as np

    tuned = _comparable(summary)
    base = dict(_comparable(baseline))
    labels = [name for name, _ in tuned]

    fig, ax = plt.subplots(figsize=(7.6, 0.66 * len(labels) + 1.9))
    y = np.arange(len(labels))

    for yi, (name, after) in zip(y, tuned):
        before = base.get(name, 0.0)
        ax.plot([before, after], [yi, yi], color=AXIS, linewidth=2,
                solid_capstyle="round", zorder=2)
        # Both ends are labelled: the light shade is below 3:1 on the
        # surface, so identity must not rest on color.
        ax.scatter([before], [yi], s=110, color=BLUE_LIGHT, zorder=3,
                   edgecolors=SURFACE, linewidths=2)
        ax.scatter([after], [yi], s=110, color=BLUE, zorder=4,
                   edgecolors=SURFACE, linewidths=2)
        left, right = min(before, after), max(before, after)
        ax.text(left - 0.02, yi, f"{before:.2f}", va="center", ha="right",
                fontsize=8.5, color=INK_MUTED)
        ax.text(right + 0.02, yi, f"{after:.2f}", va="center", ha="left",
                fontsize=8.5, color=INK_SECONDARY)

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9.5, color=INK_SECONDARY)
    ax.set_xlim(-0.14, 1.16)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["0", "0.25", "0.50", "0.75", "1.00"], fontsize=9)
    ax.invert_yaxis()
    _style_axes(ax, "x")

    handles = [
        plt.Line2D([], [], marker="o", linestyle="", markersize=9,
                   markerfacecolor=BLUE_LIGHT, markeredgecolor=SURFACE,
                   label="Base Qwen2.5-3B-Instruct"),
        plt.Line2D([], [], marker="o", linestyle="", markersize=9,
                   markerfacecolor=BLUE, markeredgecolor=SURFACE,
                   label="+ LoRA fine-tune"),
    ]
    legend = ax.legend(handles=handles, frameon=False, ncol=2, loc="upper center",
                       bbox_to_anchor=(0.5, 1.14), fontsize=9.5)
    for text in legend.get_texts():
        text.set_color(INK_SECONDARY)

    ax.set_title("Effect of fine-tuning, same 80 held-out examples",
                 fontsize=12, color=INK, pad=34, loc="left", fontweight="semibold")
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path


# --------------------------------------------------------------------------
# Runner
# --------------------------------------------------------------------------


def newest_summary(results_dir: str) -> Dict:
    matches = sorted(glob.glob(os.path.join(results_dir, "summary-*.json")))
    if not matches:
        raise FileNotFoundError(
            f"no summary-*.json in {results_dir} — run "
            f"`python -m tests.evaluation_suite.evaluate` first"
        )
    with open(matches[-1], "r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="tests/evaluation_suite/results")
    parser.add_argument("--baseline", default=None)
    parser.add_argument("--dpi", type=int, default=200)
    args = parser.parse_args()

    summary = newest_summary(args.results)
    out_dir = os.path.join(args.results, "figures")
    os.makedirs(out_dir, exist_ok=True)

    written = [
        fig_headline(summary, os.path.join(out_dir, "fig1-headline.png"), args.dpi),
        fig_per_class(summary, os.path.join(out_dir, "fig2-per-class.png"), args.dpi),
        fig_confusion(summary, os.path.join(out_dir, "fig3-confusion.png"), args.dpi),
        fig_rouge(summary, os.path.join(out_dir, "fig4-rouge.png"), args.dpi),
    ]

    if args.baseline:
        baseline = newest_summary(args.baseline)
        written.append(fig_baseline(summary, baseline,
                                    os.path.join(out_dir, "fig5-baseline.png"), args.dpi))
    else:
        print("No --baseline given; skipping the base-vs-fine-tuned figure.\n")

    for path in written:
        if path:
            print(f"wrote {path}")


if __name__ == "__main__":
    main()
