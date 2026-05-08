"""Plot train and validation loss curves from saved metrics JSONL files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot train/validation loss curves from metrics JSONL files.")
    parser.add_argument("metrics_files", nargs="+", type=Path, help="One or more metrics.jsonl files.")
    parser.add_argument("--labels", nargs="*", default=None, help="Optional labels matching metrics files.")
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("artifacts/plots/training_curves.png"),
        help="Image path for the saved plot.",
    )
    parser.add_argument("--title", type=str, default="Train and Validation Loss")
    return parser.parse_args()


def load_curves(path: Path) -> dict[str, list[tuple[int, float]]]:
    """Read per-batch train losses and validation points from one metrics file."""

    train_points: list[tuple[int, float]] = []
    val_points: list[tuple[int, float]] = []

    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            event = record.get("event")
            if event == "batch_end":
                train_points.append((int(record["global_step"]), float(record["train_loss"])))
            elif event in {"step_eval", "epoch_end"} and "val_loss" in record:
                val_points.append((int(record["global_step"]), float(record["val_loss"])))

    if not train_points:
        raise ValueError(f"No batch_end train loss records found in {path}")
    if not val_points:
        raise ValueError(f"No validation loss records found in {path}")
    return {"train": train_points, "val": val_points}


def main() -> None:
    args = parse_args()
    labels = args.labels or [path.parent.name for path in args.metrics_files]
    if len(labels) != len(args.metrics_files):
        raise ValueError("--labels must have the same number of entries as metrics_files")

    fig, ax = plt.subplots(figsize=(10, 5.5))
    colors = ["#2563eb", "#f97316", "#16a34a", "#9333ea"]

    for index, (metrics_file, label) in enumerate(zip(args.metrics_files, labels)):
        curves = load_curves(metrics_file)
        color = colors[index % len(colors)]
        train_steps, train_losses = zip(*curves["train"])
        val_steps, val_losses = zip(*curves["val"])
        ax.plot(train_steps, train_losses, linewidth=0.8, alpha=0.55, color=color, label=f"{label} train")
        ax.plot(
            val_steps,
            val_losses,
            marker="o",
            linewidth=1.8,
            color=color,
            label=f"{label} val",
        )

    ax.set_xlabel("Training Batch")
    ax.set_ylabel("Masked LM Loss")
    ax.set_title(args.title)
    ax.grid(True, alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False)
    fig.tight_layout()
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_path, dpi=180)
    print(f"Saved train/validation loss plot to {args.output_path}")


if __name__ == "__main__":
    main()
