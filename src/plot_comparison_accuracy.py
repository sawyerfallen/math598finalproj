"""Plot accuracy from comparison JSONL records."""

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
    """Plot the primary comparison metric."""

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    values = [accuracy(records, model_key, "exact_match") for model_key in MODEL_KEYS]
    bars = ax.bar(
        [MODEL_LABELS[model_key] for model_key in MODEL_KEYS],
        values,
        color=[MODEL_COLORS[model_key] for model_key in MODEL_KEYS],
        edgecolor="#111827",
        linewidth=0.5,
    )
    label_bars(ax, bars)
    ax.set_ylabel("Accuracy")
    ax.set_title(f"Overall Accuracy (n={len(records)})")
    style_axes(ax)
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
    correct_values = [accuracy(records, model_key, "exact_match") for model_key in MODEL_KEYS]
    incorrect_values = [1.0 - value for value in correct_values]

    correct_bars = ax.bar(
        list(x_positions),
        correct_values,
        color="#16a34a",
        label="Correct",
        edgecolor="#111827",
        linewidth=0.5,
    )
    ax.bar(
        list(x_positions),
        incorrect_values,
        bottom=correct_values,
        color="#d1d5db",
        label="Incorrect",
        edgecolor="#111827",
        linewidth=0.5,
    )

    label_bars(ax, correct_bars)
    ax.set_xticks(list(x_positions))
    ax.set_xticklabels([MODEL_LABELS[model_key] for model_key in MODEL_KEYS])
    ax.set_ylabel("Share of Test Set")
    ax.set_title(f"Accuracy Split (n={len(records)})")
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


def main() -> None:
    args = parse_args()
    records = load_records(args.comparison_file)
    output_dir = args.output_dir

    overall_path = output_dir / f"{args.prefix}-overall.png"
    prefix_path = output_dir / f"{args.prefix}-prefix-before-junk.png"
    by_difficulty_path = output_dir / f"{args.prefix}-by-difficulty.png"
    by_solve_kind_path = output_dir / f"{args.prefix}-by-solve-kind.png"
    split_path = output_dir / f"{args.prefix}-split.png"

    plot_overall_accuracy(records, overall_path)
    plot_metric_comparison(
        records,
        "prefix_exact_match",
        "Prefix-Before-Junk Accuracy",
        "Accuracy",
        prefix_path,
    )
    plot_metric_by_difficulty(
        records,
        "exact_match",
        "Accuracy by Difficulty",
        "Accuracy",
        by_difficulty_path,
    )
    plot_metric_by_solve_kind(
        records,
        "exact_match",
        "Accuracy by Solve Type",
        "Accuracy",
        by_solve_kind_path,
    )
    plot_correctness_stack(records, split_path)

    print(f"Saved overall accuracy plot to {overall_path}")
    print(f"Saved prefix-before-junk accuracy plot to {prefix_path}")
    print(f"Saved easy/hard accuracy plot to {by_difficulty_path}")
    print(f"Saved solve-kind accuracy plot to {by_solve_kind_path}")
    print(f"Saved accuracy split plot to {split_path}")


if __name__ == "__main__":
    main()
