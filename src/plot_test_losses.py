"""Plot per-sample test losses saved by the training scripts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot per-sample test losses from a JSONL file.")
    parser.add_argument("loss_file", type=Path, help="Path to test_sample_losses.jsonl")
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help="Optional image path. Defaults to a PNG beside the loss file.",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="Per-Sample Test Loss",
        help="Plot title.",
    )
    return parser.parse_args()


def load_loss_records(path: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            records.append(json.loads(line))
    return records


def main() -> None:
    args = parse_args()
    records = load_loss_records(args.loss_file)
    if not records:
        raise ValueError(f"No loss records found in {args.loss_file}")

    output_path = args.output_path or args.loss_file.with_suffix(".png")
    sample_indices = [int(record["sample_index"]) for record in records]
    losses = [float(record["loss"]) for record in records]

    plt.figure(figsize=(10, 5))
    plt.plot(sample_indices, losses, linewidth=1.0)
    plt.xlabel("Test Sample Index")
    plt.ylabel("Average Masked Token Loss")
    plt.title(args.title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    print(f"Saved plot to {output_path}")


if __name__ == "__main__":
    main()
