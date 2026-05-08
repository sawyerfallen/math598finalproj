"""Plot exact-match and symbolic accuracy from comparison JSONL records."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


MODEL_KEYS = ("baseline", "structured")
MODEL_LABELS = {"baseline": "Baseline", "structured": "Structured"}
MODEL_COLORS = {"baseline": "#2563eb", "structured": "#f97316"}
DIFFICULTY_ORDER = ("easy", "hard")
SOLVE_KIND_ORDER = (
    "linear_easy",
    "parenthesized_linear",
    "collect_like_terms",
    "x_both_sides",
    "distribution_both_sides",
    "quadratic_two_real_roots",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot model accuracies from comparison per-sample JSONL.")
    parser.add_argument(
        "comparison_file",
        type=Path,
        help="Path to the per-sample comparison JSONL file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/plots"),
        help="Directory where PNG plots will be saved.",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="gpt2-small-comparison",
        help="Filename prefix for generated plots.",
    )
    return parser.parse_args()


def load_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            records.append(json.loads(line))
    if not records:
        raise ValueError(f"No comparison records found in {path}")
    return records


def accuracy(records: list[dict[str, Any]], model_key: str, metric_key: str) -> float:
    correct = sum(1 for record in records if bool(record[model_key][metric_key]))
    return correct / len(records)


def grouped_by_difficulty(records: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Group comparison records by sidecar difficulty metadata."""

    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        difficulty = record.get("difficulty")
        if difficulty is None and isinstance(record.get("metadata"), dict):
            difficulty = record["metadata"].get("difficulty")
        buckets[str(difficulty or "unknown")].append(record)
    return dict(buckets)


def grouped_by_solve_kind(records: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Group comparison records by generated equation subtype."""

    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        solve_kind = record.get("solve_kind")
        if solve_kind is None and isinstance(record.get("metadata"), dict):
            solve_kind = record["metadata"].get("solve_kind")
        buckets[str(solve_kind or "unknown")].append(record)
    return dict(buckets)


def sorted_difficulties(difficulty_records: dict[str, list[dict[str, Any]]]) -> list[str]:
    known = [difficulty for difficulty in DIFFICULTY_ORDER if difficulty in difficulty_records]
    extra = sorted(difficulty for difficulty in difficulty_records if difficulty not in DIFFICULTY_ORDER)
    return known + extra


def sorted_solve_kinds(solve_kind_records: dict[str, list[dict[str, Any]]]) -> list[str]:
    known = [solve_kind for solve_kind in SOLVE_KIND_ORDER if solve_kind in solve_kind_records]
    extra = sorted(solve_kind for solve_kind in solve_kind_records if solve_kind not in SOLVE_KIND_ORDER)
    return known + extra


def style_axes(ax: plt.Axes) -> None:
    ax.set_ylim(0, 1.0)
    ax.yaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
    ax.grid(axis="y", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def label_bars(ax: plt.Axes, bars: Any) -> None:
    for bar in bars:
        height = bar.get_height()
        # Keep zero labels visible just above the axis line.
        y = height + 0.015 if height > 0 else 0.02
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            y,
            f"{height:.1%}",
            ha="center",
            va="bottom",
            fontsize=9,
            color="#111827",
        )


def plot_overall_accuracy(records: list[dict[str, Any]], output_path: Path) -> None:
    metric_labels = ("Exact Match", "Symbolic")
    metric_keys = ("exact_match", "symbolic_match")
    x_positions = range(len(metric_keys))
    width = 0.34

    fig, ax = plt.subplots(figsize=(8, 5))
    for offset_index, model_key in enumerate(MODEL_KEYS):
        values = [accuracy(records, model_key, metric_key) for metric_key in metric_keys]
        offset = (offset_index - 0.5) * width
        bars = ax.bar(
            [x + offset for x in x_positions],
            values,
            width=width,
            label=MODEL_LABELS[model_key],
            color=MODEL_COLORS[model_key],
            edgecolor="#111827",
            linewidth=0.5,
        )
        label_bars(ax, bars)

    ax.set_xticks(list(x_positions))
    ax.set_xticklabels(metric_labels)
    ax.set_ylabel("Accuracy")
    ax.set_title(f"Overall Test Accuracy (n={len(records)})")
    style_axes(ax)
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_metric_by_difficulty(
    records: list[dict[str, Any]],
    metric_key: str,
    title: str,
    ylabel: str,
    output_path: Path,
) -> None:
    """Plot baseline/structured metric values separately for easy and hard examples."""

    difficulty_records = grouped_by_difficulty(records)
    difficulties = sorted_difficulties(difficulty_records)
    x_positions = range(len(difficulties))
    width = 0.36

    fig, ax = plt.subplots(figsize=(8, 5))
    for offset_index, model_key in enumerate(MODEL_KEYS):
        values = [accuracy(difficulty_records[difficulty], model_key, metric_key) for difficulty in difficulties]
        offset = (offset_index - 0.5) * width
        bars = ax.bar(
            [x + offset for x in x_positions],
            values,
            width=width,
            label=MODEL_LABELS[model_key],
            color=MODEL_COLORS[model_key],
            edgecolor="#111827",
            linewidth=0.5,
        )
        label_bars(ax, bars)

    labels = [f"{difficulty}\n(n={len(difficulty_records[difficulty])})" for difficulty in difficulties]
    ax.set_xticks(list(x_positions))
    ax.set_xticklabels(labels)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    style_axes(ax)
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_metric_by_solve_kind(
    records: list[dict[str, Any]],
    metric_key: str,
    title: str,
    ylabel: str,
    output_path: Path,
) -> None:
    """Plot baseline/structured metric values for each generated solve subtype."""

    solve_kind_records = grouped_by_solve_kind(records)
    solve_kinds = sorted_solve_kinds(solve_kind_records)
    x_positions = range(len(solve_kinds))
    width = 0.36

    fig, ax = plt.subplots(figsize=(12, 5.5))
    for offset_index, model_key in enumerate(MODEL_KEYS):
        values = [accuracy(solve_kind_records[solve_kind], model_key, metric_key) for solve_kind in solve_kinds]
        offset = (offset_index - 0.5) * width
        bars = ax.bar(
            [x + offset for x in x_positions],
            values,
            width=width,
            label=MODEL_LABELS[model_key],
            color=MODEL_COLORS[model_key],
            edgecolor="#111827",
            linewidth=0.5,
        )
        label_bars(ax, bars)

    labels = [
        f"{solve_kind.replace('_', ' ')}\n(n={len(solve_kind_records[solve_kind])})"
        for solve_kind in solve_kinds
    ]
    ax.set_xticks(list(x_positions))
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    style_axes(ax)
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_correctness_stack(records: list[dict[str, Any]], output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 5))
    x_positions = range(len(MODEL_KEYS))
    correct_values = [accuracy(records, model_key, "symbolic_match") for model_key in MODEL_KEYS]
    incorrect_values = [1.0 - value for value in correct_values]

    correct_bars = ax.bar(
        list(x_positions),
        correct_values,
        color="#16a34a",
        label="Symbolically Correct",
        edgecolor="#111827",
        linewidth=0.5,
    )
    ax.bar(
        list(x_positions),
        incorrect_values,
        bottom=correct_values,
        color="#d1d5db",
        label="Symbolically Incorrect",
        edgecolor="#111827",
        linewidth=0.5,
    )

    label_bars(ax, correct_bars)
    ax.set_xticks(list(x_positions))
    ax.set_xticklabels([MODEL_LABELS[model_key] for model_key in MODEL_KEYS])
    ax.set_ylabel("Share of Test Set")
    ax.set_title(f"Symbolic Correctness Split (n={len(records)})")
    style_axes(ax)
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_metric_comparison(
    records: list[dict[str, Any]],
    metric_key: str,
    title: str,
    ylabel: str,
    output_path: Path,
) -> None:
    """Plot one accuracy-like metric for baseline vs structured."""

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    values = [accuracy(records, model_key, metric_key) for model_key in MODEL_KEYS]
    bars = ax.bar(
        [MODEL_LABELS[model_key] for model_key in MODEL_KEYS],
        values,
        color=[MODEL_COLORS[model_key] for model_key in MODEL_KEYS],
        edgecolor="#111827",
        linewidth=0.5,
    )
    label_bars(ax, bars)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    style_axes(ax)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_generation_breakdown(records: list[dict[str, Any]], output_path: Path) -> None:
    """Show how often outputs were clean, salvaged from a prefix, or still wrong."""

    fig, ax = plt.subplots(figsize=(8, 5))
    x_positions = range(len(MODEL_KEYS))
    clean_correct = []
    prefix_correct = []
    incorrect = []

    for model_key in MODEL_KEYS:
        clean = sum(
            1
            for record in records
            if record[model_key]["symbolic_match"] and record[model_key]["stopped_cleanly"]
        ) / len(records)
        prefix = sum(1 for record in records if record[model_key]["prefix_symbolic_match"]) / len(records)
        clean_correct.append(clean)
        prefix_correct.append(prefix)
        incorrect.append(max(0.0, 1.0 - clean - prefix))

    ax.bar(
        list(x_positions),
        clean_correct,
        color="#16a34a",
        label="Clean Correct",
        edgecolor="#111827",
        linewidth=0.5,
    )
    ax.bar(
        list(x_positions),
        prefix_correct,
        bottom=clean_correct,
        color="#22c55e",
        label="Correct Prefix Before Junk",
        edgecolor="#111827",
        linewidth=0.5,
    )
    bottoms = [clean + prefix for clean, prefix in zip(clean_correct, prefix_correct)]
    ax.bar(
        list(x_positions),
        incorrect,
        bottom=bottoms,
        color="#d1d5db",
        label="Incorrect",
        edgecolor="#111827",
        linewidth=0.5,
    )

    ax.set_xticks(list(x_positions))
    ax.set_xticklabels([MODEL_LABELS[model_key] for model_key in MODEL_KEYS])
    ax.set_ylabel("Share of Test Set")
    ax.set_title(f"Generation Outcome Breakdown (n={len(records)})")
    style_axes(ax)
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    records = load_records(args.comparison_file)
    output_dir = args.output_dir

    overall_path = output_dir / f"{args.prefix}-overall-accuracy.png"
    exact_path = output_dir / f"{args.prefix}-exact-accuracy.png"
    symbolic_path = output_dir / f"{args.prefix}-symbolic-accuracy.png"
    prefix_path = output_dir / f"{args.prefix}-prefix-before-junk-accuracy.png"
    exact_by_difficulty_path = output_dir / f"{args.prefix}-exact-accuracy-by-difficulty.png"
    symbolic_by_difficulty_path = output_dir / f"{args.prefix}-symbolic-accuracy-by-difficulty.png"
    exact_by_solve_kind_path = output_dir / f"{args.prefix}-exact-accuracy-by-solve-kind.png"
    symbolic_by_solve_kind_path = output_dir / f"{args.prefix}-symbolic-accuracy-by-solve-kind.png"
    stack_path = output_dir / f"{args.prefix}-symbolic-correctness-stack.png"
    breakdown_path = output_dir / f"{args.prefix}-generation-breakdown.png"

    plot_overall_accuracy(records, overall_path)
    plot_metric_comparison(records, "exact_match", "Exact-Match Accuracy", "Exact-Match Accuracy", exact_path)
    plot_metric_comparison(records, "symbolic_match", "Symbolic Accuracy", "Symbolic Accuracy", symbolic_path)
    plot_metric_comparison(
        records,
        "prefix_symbolic_match",
        "Correct Prefix Before Junk",
        "Share of Test Set",
        prefix_path,
    )
    plot_metric_by_difficulty(
        records,
        "exact_match",
        "Exact-Match Accuracy by Difficulty",
        "Exact-Match Accuracy",
        exact_by_difficulty_path,
    )
    plot_metric_by_difficulty(
        records,
        "symbolic_match",
        "Symbolic Accuracy by Difficulty",
        "Symbolic Accuracy",
        symbolic_by_difficulty_path,
    )
    plot_metric_by_solve_kind(
        records,
        "exact_match",
        "Exact-Match Accuracy by Solve Type",
        "Exact-Match Accuracy",
        exact_by_solve_kind_path,
    )
    plot_metric_by_solve_kind(
        records,
        "symbolic_match",
        "Symbolic Accuracy by Solve Type",
        "Symbolic Accuracy",
        symbolic_by_solve_kind_path,
    )
    plot_correctness_stack(records, stack_path)
    plot_generation_breakdown(records, breakdown_path)

    print(f"Saved overall accuracy plot to {overall_path}")
    print(f"Saved exact-match accuracy plot to {exact_path}")
    print(f"Saved symbolic accuracy plot to {symbolic_path}")
    print(f"Saved correct-prefix plot to {prefix_path}")
    print(f"Saved easy/hard exact-match plot to {exact_by_difficulty_path}")
    print(f"Saved easy/hard symbolic plot to {symbolic_by_difficulty_path}")
    print(f"Saved solve-kind exact-match plot to {exact_by_solve_kind_path}")
    print(f"Saved solve-kind symbolic plot to {symbolic_by_solve_kind_path}")
    print(f"Saved symbolic correctness stack plot to {stack_path}")
    print(f"Saved generation breakdown plot to {breakdown_path}")


if __name__ == "__main__":
    main()
