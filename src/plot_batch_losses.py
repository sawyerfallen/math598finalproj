"""Plot training loss over batches for one or more metrics JSONL files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot batch training losses from metrics.jsonl files.")
    parser.add_argument("metrics_files", nargs="+", type=Path, help="One or more metrics.jsonl files.")
    parser.add_argument(
        "--labels",
        nargs="*",
        default=None,
        help="Optional display labels matching the metrics files.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("artifacts/plots/batch_losses.png"),
        help="Image path for the saved plot.",
    )
    parser.add_argument("--title", type=str, default="Training Loss Over Batches")
    return parser.parse_args()


def load_batch_losses(path: Path) -> tuple[list[int], list[float]]:
    """Read the per-batch records emitted during training."""

    steps: list[int] = []
    losses: list[float] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if record.get("event") != "batch_end":
                continue
            steps.append(int(record["global_step"]))
            losses.append(float(record["train_loss"]))
    if not losses:
        raise ValueError(f"No batch_end records found in {path}")
    return steps, losses


def main() -> None:
    args = parse_args()
    labels = args.labels or [path.parent.name for path in args.metrics_files]
    if len(labels) != len(args.metrics_files):
        raise ValueError("--labels must have the same number of entries as metrics_files")

    plt.figure(figsize=(10, 5))
    for metrics_file, label in zip(args.metrics_files, labels):
        steps, losses = load_batch_losses(metrics_file)
        plt.plot(steps, losses, linewidth=1.0, label=label)

    plt.xlabel("Training Batch")
    plt.ylabel("Masked LM Loss")
    plt.title(args.title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.output_path, dpi=150)
    print(f"Saved plot to {args.output_path}")


if __name__ == "__main__":
    main()
