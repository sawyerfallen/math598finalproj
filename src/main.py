"""Compare saved baseline and structured checkpoints on algebra generation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from .algebra_generation import (
        build_allowed_token_mask,
        generate_baseline_predictions,
        generate_structured_predictions,
    )
    from .structured_model import StructuredCausalLM
    from .train_baseline import JsonlAlgebraDataset, is_symbolically_equivalent
    from .utils import ensure_padding_token
except ImportError:
    from algebra_generation import (
        build_allowed_token_mask,
        generate_baseline_predictions,
        generate_structured_predictions,
    )
    from structured_model import StructuredCausalLM
    from train_baseline import JsonlAlgebraDataset, is_symbolically_equivalent
    from utils import ensure_padding_token


DEFAULT_BASELINE_CHECKPOINT = Path("artifacts/models_training_info/gpt2-small-baseline/final-model")
DEFAULT_STRUCTURED_CHECKPOINT = Path("artifacts/models_training_info/gpt2-small-structured-node-types/final-model")
DEFAULT_OUTPUT_PATH = Path("artifacts/comparisons/gpt2_small_baseline_vs_structured.json")
DEFAULT_TEXT_OUTPUT_PATH = Path("artifacts/comparisons/gpt2_small_baseline_vs_structured.txt")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare saved baseline and structured GPT-2 small checkpoints.")
    parser.add_argument("--baseline-checkpoint", type=Path, default=DEFAULT_BASELINE_CHECKPOINT)
    parser.add_argument("--structured-checkpoint", type=Path, default=DEFAULT_STRUCTURED_CHECKPOINT)
    parser.add_argument("--test-path", type=Path, default=Path("data/test.jsonl"))
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--text-output-path", type=Path, default=DEFAULT_TEXT_OUTPUT_PATH)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_examples(path: Path, max_samples: int | None) -> JsonlAlgebraDataset:
    """Load the evaluation split and optionally keep only a prefix for quick runs."""

    dataset = JsonlAlgebraDataset(path)
    return dataset.take(max_samples)


def load_baseline_model(checkpoint: Path, device: torch.device) -> tuple[Any, torch.nn.Module, torch.Tensor]:
    """Load the baseline tokenizer/model pair and build its constrained decoding mask."""

    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    tokenizer_vocab_grew = ensure_padding_token(tokenizer)
    model = AutoModelForCausalLM.from_pretrained(checkpoint, torch_dtype=torch.float32).to(device)
    if tokenizer_vocab_grew or len(tokenizer) != model.get_input_embeddings().num_embeddings:
        model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = tokenizer.pad_token_id
    allowed_token_mask = build_allowed_token_mask(tokenizer, device, vocab_size=model.config.vocab_size)
    return tokenizer, model, allowed_token_mask


def load_structured_model(checkpoint: Path, device: torch.device) -> tuple[Any, StructuredCausalLM, torch.Tensor]:
    """Load the structured wrapper and keep tokenizer/model vocabulary sizes aligned."""

    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    tokenizer_vocab_grew = ensure_padding_token(tokenizer)
    model = StructuredCausalLM.from_checkpoint(checkpoint, freeze_base=True).to(device)
    if tokenizer_vocab_grew or len(tokenizer) != model.base.get_input_embeddings().num_embeddings:
        model.base.resize_token_embeddings(len(tokenizer))
        for parameter in model.base.parameters():
            parameter.requires_grad = False
    model.base.config.pad_token_id = tokenizer.pad_token_id
    allowed_token_mask = build_allowed_token_mask(tokenizer, device, vocab_size=model.base.config.vocab_size)
    return tokenizer, model, allowed_token_mask


def compute_accuracy_metrics(predictions: list[str], dataset: JsonlAlgebraDataset) -> dict[str, Any]:
    """Compute exact-string and SymPy-based symbolic accuracies."""

    exact_matches = 0
    symbolic_matches = 0
    mistakes = []

    for prediction, example in zip(predictions, dataset.examples):
        target = example.output.strip()
        exact_match = prediction == target
        symbolic_match = is_symbolically_equivalent(prediction, target)

        if exact_match:
            exact_matches += 1
        if symbolic_match:
            symbolic_matches += 1

        if len(mistakes) < 5 and not exact_match:
            mistakes.append(
                {
                    "prompt": example.prompt,
                    "target": target,
                    "prediction": prediction,
                    "symbolically_correct": symbolic_match,
                }
            )

    total = max(len(dataset), 1)
    return {
        "num_examples": len(dataset),
        "exact_match_accuracy": exact_matches / total,
        "symbolic_accuracy": symbolic_matches / total,
        "sample_mistakes": mistakes,
    }


def print_metrics(label: str, metrics: dict[str, Any]) -> None:
    print(label)
    print(f"  examples: {metrics['num_examples']}")
    print(f"  exact match accuracy: {metrics['exact_match_accuracy']:.4f}")
    print(f"  symbolic accuracy: {metrics['symbolic_accuracy']:.4f}")


def build_comparison_table(
    baseline_metrics: dict[str, Any],
    structured_metrics: dict[str, Any],
) -> str:
    rows = [
        (
            "Metric",
            "Baseline",
            "Structured",
            "Delta",
        ),
        (
            "Exact Match Accuracy",
            f"{baseline_metrics['exact_match_accuracy']:.4f}",
            f"{structured_metrics['exact_match_accuracy']:.4f}",
            f"{structured_metrics['exact_match_accuracy'] - baseline_metrics['exact_match_accuracy']:+.4f}",
        ),
        (
            "Symbolic Accuracy",
            f"{baseline_metrics['symbolic_accuracy']:.4f}",
            f"{structured_metrics['symbolic_accuracy']:.4f}",
            f"{structured_metrics['symbolic_accuracy'] - baseline_metrics['symbolic_accuracy']:+.4f}",
        ),
    ]

    widths = [max(len(row[column_index]) for row in rows) for column_index in range(len(rows[0]))]

    def format_row(row: tuple[str, str, str, str]) -> str:
        return " | ".join(cell.ljust(width) for cell, width in zip(row, widths))

    separator = "-+-".join("-" * width for width in widths)
    return "\n".join([format_row(rows[0]), separator, *(format_row(row) for row in rows[1:])])


def build_text_summary(
    args: argparse.Namespace,
    baseline_metrics: dict[str, Any],
    structured_metrics: dict[str, Any],
    table: str,
) -> str:
    lines = [
        "Baseline vs Structured Comparison",
        f"Baseline checkpoint: {args.baseline_checkpoint}",
        f"Structured checkpoint: {args.structured_checkpoint}",
        f"Evaluation set: {args.test_path}",
        f"Examples: {baseline_metrics['num_examples']}",
        "",
        table,
        "",
        "Sample baseline mistakes:",
    ]

    if baseline_metrics["sample_mistakes"]:
        for item in baseline_metrics["sample_mistakes"]:
            lines.extend(
                [
                    f"- Prompt: {item['prompt']}",
                    f"  Target: {item['target']}",
                    f"  Prediction: {item['prediction']}",
                    f"  Symbolically correct: {item['symbolically_correct']}",
                ]
            )
    else:
        lines.append("- None")

    lines.append("")
    lines.append("Sample structured mistakes:")
    if structured_metrics["sample_mistakes"]:
        for item in structured_metrics["sample_mistakes"]:
            lines.extend(
                [
                    f"- Prompt: {item['prompt']}",
                    f"  Target: {item['target']}",
                    f"  Prediction: {item['prediction']}",
                    f"  Symbolically correct: {item['symbolically_correct']}",
                ]
            )
    else:
        lines.append("- None")

    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    if not args.baseline_checkpoint.exists():
        raise FileNotFoundError(f"Baseline checkpoint not found: {args.baseline_checkpoint}")
    if not args.structured_checkpoint.exists():
        raise FileNotFoundError(f"Structured checkpoint not found: {args.structured_checkpoint}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dataset = load_examples(args.test_path, args.max_samples)
    print(f"Loaded {len(dataset)} comparison examples from {args.test_path}")

    baseline_tokenizer, baseline_model, baseline_allowed_token_mask = load_baseline_model(
        args.baseline_checkpoint,
        device,
    )
    structured_tokenizer, structured_model, structured_allowed_token_mask = load_structured_model(
        args.structured_checkpoint,
        device,
    )

    baseline_predictions: list[str] = []
    structured_predictions: list[str] = []

    for start in range(0, len(dataset), args.batch_size):
        prompts = [example.prompt for example in dataset.examples[start:start + args.batch_size]]
        baseline_predictions.extend(
            generate_baseline_predictions(
                model=baseline_model,
                tokenizer=baseline_tokenizer,
                prompts=prompts,
                device=device,
                max_new_tokens=args.max_new_tokens,
                allowed_token_mask=baseline_allowed_token_mask,
            )
        )
        structured_predictions.extend(
            generate_structured_predictions(
                model=structured_model,
                tokenizer=structured_tokenizer,
                prompts=prompts,
                device=device,
                max_new_tokens=args.max_new_tokens,
                allowed_token_mask=structured_allowed_token_mask,
            )
        )

    baseline_metrics = compute_accuracy_metrics(baseline_predictions, dataset)
    structured_metrics = compute_accuracy_metrics(structured_predictions, dataset)
    table = build_comparison_table(baseline_metrics, structured_metrics)

    print_metrics("Baseline", baseline_metrics)
    print_metrics("Structured", structured_metrics)
    print("")
    print(table)

    comparison = {
        "baseline_checkpoint": str(args.baseline_checkpoint),
        "structured_checkpoint": str(args.structured_checkpoint),
        "test_path": str(args.test_path),
        "max_samples": args.max_samples,
        "batch_size": args.batch_size,
        "max_new_tokens": args.max_new_tokens,
        "baseline": baseline_metrics,
        "structured": structured_metrics,
        "accuracy_delta": {
            "exact_match_accuracy": structured_metrics["exact_match_accuracy"]
            - baseline_metrics["exact_match_accuracy"],
            "symbolic_accuracy": structured_metrics["symbolic_accuracy"]
            - baseline_metrics["symbolic_accuracy"],
        },
    }

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_path.write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    args.text_output_path.parent.mkdir(parents=True, exist_ok=True)
    args.text_output_path.write_text(
        build_text_summary(args, baseline_metrics, structured_metrics, table),
        encoding="utf-8",
    )
    print(f"Saved comparison to {args.output_path}")
    print(f"Saved text comparison to {args.text_output_path}")


if __name__ == "__main__":
    main()
