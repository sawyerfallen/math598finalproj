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
TASK_ORDER = ("simplify", "expand", "factor", "solve", "substitute")


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


def task_name(prompt: str) -> str:
    """Use the first prompt word as the algebra task bucket."""

    return prompt.split(maxsplit=1)[0].strip().lower() or "unknown"


def accuracy(records: list[dict[str, Any]], model_key: str, metric_key: str) -> float:
    correct = sum(1 for record in records if bool(record[model_key][metric_key]))
    return correct / len(records)


def grouped_by_task(records: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        buckets[task_name(str(record["prompt"]))].append(record)
    return dict(buckets)


def sorted_tasks(task_records: dict[str, list[dict[str, Any]]]) -> list[str]:
    known_tasks = [task for task in TASK_ORDER if task in task_records]
    extra_tasks = sorted(task for task in task_records if task not in TASK_ORDER)
    return known_tasks + extra_tasks


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


def plot_symbolic_accuracy_by_task(records: list[dict[str, Any]], output_path: Path) -> None:
    task_records = grouped_by_task(records)
    tasks = sorted_tasks(task_records)
    x_positions = range(len(tasks))
    width = 0.36

    fig, ax = plt.subplots(figsize=(11, 5.5))
    for offset_index, model_key in enumerate(MODEL_KEYS):
        values = [accuracy(task_records[task], model_key, "symbolic_match") for task in tasks]
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

    labels = [f"{task}\n(n={len(task_records[task])})" for task in tasks]
    ax.set_xticks(list(x_positions))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Symbolic Accuracy")
    ax.set_title("Symbolic Accuracy by Task")
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


def main() -> None:
    args = parse_args()
    records = load_records(args.comparison_file)
    output_dir = args.output_dir

    overall_path = output_dir / f"{args.prefix}-overall-accuracy.png"
    by_task_path = output_dir / f"{args.prefix}-symbolic-accuracy-by-task.png"
    stack_path = output_dir / f"{args.prefix}-symbolic-correctness-stack.png"

    plot_overall_accuracy(records, overall_path)
    plot_symbolic_accuracy_by_task(records, by_task_path)
    plot_correctness_stack(records, stack_path)

    print(f"Saved overall accuracy plot to {overall_path}")
    print(f"Saved per-task symbolic accuracy plot to {by_task_path}")
    print(f"Saved symbolic correctness stack plot to {stack_path}")


if __name__ == "__main__":
    main()
