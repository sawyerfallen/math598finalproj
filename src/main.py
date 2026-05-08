"""Compare saved baseline and structured checkpoints on algebra generation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .algebra_generation import (
    build_allowed_token_mask,
    extract_first_answer_span,
    generate_baseline_predictions,
    generate_structured_predictions,
    postprocess_prediction,
)
from .structured_model import StructuredCausalLM
from .train_baseline import JsonlAlgebraDataset, is_symbolically_equivalent
from .utils import ensure_padding_token


DEFAULT_BASELINE_CHECKPOINT = Path(
    "artifacts/models_training_info/gpt2-small-baseline-solve-mixed-10000-dedup/best-epoch-1"
)
DEFAULT_STRUCTURED_CHECKPOINT = Path(
    "artifacts/models_training_info/gpt2-small-structured-solve-mixed-10000-dedup/best-epoch-1"
)
DEFAULT_TEST_PATH = Path("data/solve_mixed_10000_dedup/test.jsonl")
DEFAULT_METADATA_PATH = Path("data/solve_mixed_10000_dedup/test_metadata.jsonl")
DEFAULT_OUTPUT_PATH = Path("artifacts/comparisons/gpt2-small-solve-mixed-10000-dedup-baseline-vs-structured.json")
DEFAULT_TEXT_OUTPUT_PATH = Path("artifacts/comparisons/gpt2-small-solve-mixed-10000-dedup-baseline-vs-structured.txt")
DEFAULT_PER_SAMPLE_OUTPUT_PATH = Path(
    "artifacts/comparisons/gpt2-small-solve-mixed-10000-dedup-baseline-vs-structured-per-sample.jsonl"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare saved baseline and structured GPT-2 small checkpoints.")
    parser.add_argument("--baseline-checkpoint", type=Path, default=DEFAULT_BASELINE_CHECKPOINT)
    parser.add_argument("--structured-checkpoint", type=Path, default=DEFAULT_STRUCTURED_CHECKPOINT)
    parser.add_argument("--test-path", type=Path, default=DEFAULT_TEST_PATH)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--text-output-path", type=Path, default=DEFAULT_TEXT_OUTPUT_PATH)
    parser.add_argument("--per-sample-output-path", type=Path, default=DEFAULT_PER_SAMPLE_OUTPUT_PATH)
    parser.add_argument(
        "--metadata-path",
        type=Path,
        default=DEFAULT_METADATA_PATH,
        help="Optional split metadata JSONL with difficulty/solve_kind fields for grouped evaluation.",
    )
    parser.add_argument(
        "--no-stop-on-valid-answer",
        action="store_true",
        help="Keep decoding until EOS/max_new_tokens instead of stopping once a complete task answer appears.",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_examples(path: Path, max_samples: int | None) -> JsonlAlgebraDataset:
    """Load the evaluation split and optionally keep only a prefix for quick runs."""

    dataset = JsonlAlgebraDataset(path)
    return dataset.take(max_samples)


def load_metadata_records(path: Path | None, max_samples: int | None) -> list[dict[str, Any]] | None:
    """Load optional sidecar metadata without changing the clean training/eval JSONL."""

    if path is None:
        return None
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            record = json.loads(line)
            if "prompt" not in record or "output" not in record:
                raise ValueError(f"{path}:{line_number} is missing 'prompt' or 'output'")
            records.append(record)
            if max_samples is not None and len(records) >= max_samples:
                break
    return records


def verify_structured_checkpoint(checkpoint: Path) -> dict[str, Any]:
    """Confirm the structured checkpoint contains learned node-type weights."""

    state_path = checkpoint / "structured_state.pt"
    verification: dict[str, Any] = {
        "structured_state_path": str(state_path),
        "structured_state_exists": state_path.exists(),
        "has_node_type_embedding": False,
        "node_type_embedding_shape": None,
        "node_type_embedding_abs_sum": None,
        "node_type_embedding_nonzero": False,
    }
    if not state_path.exists():
        return verification

    state = torch.load(state_path, map_location="cpu")
    embedding_state = state.get("node_type_embedding", {})
    weight = embedding_state.get("weight")
    if weight is None:
        return verification

    abs_sum = float(weight.abs().sum().item())
    verification.update(
        {
            "has_node_type_embedding": True,
            "node_type_embedding_shape": list(weight.shape),
            "node_type_embedding_abs_sum": abs_sum,
            "node_type_embedding_nonzero": abs_sum > 0.0,
        }
    )
    return verification


def load_baseline_model(checkpoint: Path, device: torch.device) -> tuple[Any, torch.nn.Module, torch.Tensor]:
    """Load the baseline tokenizer/model pair and build its constrained decoding mask."""

    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    tokenizer_vocab_grew = ensure_padding_token(tokenizer)
    model = AutoModelForCausalLM.from_pretrained(checkpoint, torch_dtype=torch.float32).to(device)
    if tokenizer_vocab_grew or len(tokenizer) != model.get_input_embeddings().num_embeddings:
        model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = tokenizer.pad_token_id
    model.eval()
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
    model.eval()
    allowed_token_mask = build_allowed_token_mask(tokenizer, device, vocab_size=model.base.config.vocab_size)
    return tokenizer, model, allowed_token_mask


def evaluate_prediction(prompt: str, raw_prediction: str, target: str) -> dict[str, Any]:
    """Score raw generation plus the extracted first answer span."""

    raw_prediction = postprocess_prediction(raw_prediction)
    extracted_prediction = extract_first_answer_span(prompt, raw_prediction)
    raw_exact_match = raw_prediction == target
    raw_symbolic_match = is_symbolically_equivalent(raw_prediction, target)
    exact_match = extracted_prediction == target
    symbolic_match = is_symbolically_equivalent(extracted_prediction, target)

    return {
        "raw_prediction": raw_prediction,
        "prediction": extracted_prediction,
        "raw_exact_match": raw_exact_match,
        "raw_symbolic_match": raw_symbolic_match,
        "exact_match": exact_match,
        "symbolic_match": symbolic_match,
        # This metric is useful when the raw continuation contains a correct
        # answer span followed by extra text that should not define the answer.
        "prefix_symbolic_match": symbolic_match and raw_prediction != extracted_prediction,
        "span_symbolic_match": symbolic_match,
        "had_trailing_junk": raw_prediction != extracted_prediction,
        "stopped_cleanly": raw_prediction == extracted_prediction,
    }


def compute_accuracy_metrics(records: list[dict[str, Any]], model_key: str) -> dict[str, Any]:
    """Compute raw and extracted-answer accuracies from saved per-sample records."""

    total = max(len(records), 1)
    exact_matches = sum(1 for record in records if record[model_key]["exact_match"])
    symbolic_matches = sum(1 for record in records if record[model_key]["symbolic_match"])
    raw_exact_matches = sum(1 for record in records if record[model_key]["raw_exact_match"])
    raw_symbolic_matches = sum(1 for record in records if record[model_key]["raw_symbolic_match"])
    prefix_symbolic_matches = sum(1 for record in records if record[model_key]["prefix_symbolic_match"])
    trailing_junk_count = sum(1 for record in records if record[model_key]["had_trailing_junk"])
    stopped_cleanly_count = sum(1 for record in records if record[model_key]["stopped_cleanly"])

    mistakes = []
    for record in records:
        model_record = record[model_key]
        if len(mistakes) < 5 and not model_record["exact_match"]:
            mistakes.append(
                {
                    "prompt": record["prompt"],
                    "target": record["target"],
                    "raw_prediction": model_record["raw_prediction"],
                    "prediction": model_record["prediction"],
                    "symbolically_correct": model_record["symbolic_match"],
                }
            )

    return {
        "num_examples": len(records),
        "exact_match_accuracy": exact_matches / total,
        "symbolic_accuracy": symbolic_matches / total,
        "raw_exact_match_accuracy": raw_exact_matches / total,
        "raw_symbolic_accuracy": raw_symbolic_matches / total,
        "prefix_symbolic_accuracy": prefix_symbolic_matches / total,
        "trailing_junk_rate": trailing_junk_count / total,
        "stopped_cleanly_rate": stopped_cleanly_count / total,
        "sample_mistakes": mistakes,
    }


def compute_grouped_metrics(records: list[dict[str, Any]], model_key: str, group_key: str) -> dict[str, Any]:
    """Compute the same metrics for each metadata group, such as easy vs hard."""

    grouped_records: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        group_value = record.get(group_key)
        if group_value is None and isinstance(record.get("metadata"), dict):
            group_value = record["metadata"].get(group_key)
        if group_value is None:
            group_value = "unknown"
        grouped_records.setdefault(str(group_value), []).append(record)

    return {
        group: compute_accuracy_metrics(group_records, model_key)
        for group, group_records in sorted(grouped_records.items())
    }


def build_per_sample_records(
    dataset: JsonlAlgebraDataset,
    baseline_predictions: list[str],
    structured_predictions: list[str],
    metadata_records: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Build one saved record per test example for later plotting and inspection."""

    records = []
    for index, (example, baseline_prediction, structured_prediction) in enumerate(
        zip(dataset.examples, baseline_predictions, structured_predictions)
    ):
        metadata = metadata_records[index] if metadata_records is not None else {}
        if metadata:
            if str(metadata.get("prompt")) != example.prompt or str(metadata.get("output")) != example.output:
                raise ValueError(
                    "Metadata sidecar is not aligned with the evaluation JSONL at "
                    f"index {index}: {metadata.get('prompt')!r} vs {example.prompt!r}"
                )

        target = example.output.strip()
        baseline_record = evaluate_prediction(example.prompt, baseline_prediction, target)
        structured_record = evaluate_prediction(example.prompt, structured_prediction, target)

        records.append(
            {
                "sample_index": index,
                "prompt": example.prompt,
                "target": target,
                "metadata": metadata,
                "task": metadata.get("task") if metadata else None,
                "difficulty": metadata.get("difficulty") if metadata else None,
                "solve_kind": metadata.get("solve_kind") if metadata else None,
                "baseline": baseline_record,
                "structured": structured_record,
                "comparison": {
                    "both_symbolically_correct": baseline_record["symbolic_match"]
                    and structured_record["symbolic_match"],
                    "only_baseline_symbolically_correct": baseline_record["symbolic_match"]
                    and not structured_record["symbolic_match"],
                    "only_structured_symbolically_correct": structured_record["symbolic_match"]
                    and not baseline_record["symbolic_match"],
                    "both_symbolically_wrong": not baseline_record["symbolic_match"]
                    and not structured_record["symbolic_match"],
                },
            }
        )

    return records


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    """Write structured per-example data in a plotting-friendly JSONL format."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")


def print_metrics(label: str, metrics: dict[str, Any]) -> None:
    print(label)
    print(f"  examples: {metrics['num_examples']}")
    print(f"  exact match accuracy: {metrics['exact_match_accuracy']:.4f}")
    print(f"  symbolic accuracy: {metrics['symbolic_accuracy']:.4f}")
    print(f"  raw symbolic accuracy: {metrics['raw_symbolic_accuracy']:.4f}")
    print(f"  prefix-before-junk symbolic accuracy: {metrics['prefix_symbolic_accuracy']:.4f}")


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
        (
            "Raw Symbolic Accuracy",
            f"{baseline_metrics['raw_symbolic_accuracy']:.4f}",
            f"{structured_metrics['raw_symbolic_accuracy']:.4f}",
            f"{structured_metrics['raw_symbolic_accuracy'] - baseline_metrics['raw_symbolic_accuracy']:+.4f}",
        ),
        (
            "Prefix Symbolic Accuracy",
            f"{baseline_metrics['prefix_symbolic_accuracy']:.4f}",
            f"{structured_metrics['prefix_symbolic_accuracy']:.4f}",
            f"{structured_metrics['prefix_symbolic_accuracy'] - baseline_metrics['prefix_symbolic_accuracy']:+.4f}",
        ),
    ]

    widths = [max(len(row[column_index]) for row in rows) for column_index in range(len(rows[0]))]

    def format_row(row: tuple[str, str, str, str]) -> str:
        return " | ".join(cell.ljust(width) for cell, width in zip(row, widths))

    separator = "-+-".join("-" * width for width in widths)
    return "\n".join([format_row(rows[0]), separator, *(format_row(row) for row in rows[1:])])


def build_difficulty_table(
    baseline_by_difficulty: dict[str, Any],
    structured_by_difficulty: dict[str, Any],
) -> str:
    """Format easy/hard grouped exact and symbolic accuracies."""

    difficulties = sorted(set(baseline_by_difficulty) | set(structured_by_difficulty))
    rows = [("Difficulty", "Examples", "Baseline Exact", "Baseline Symbolic", "Structured Exact", "Structured Symbolic")]
    for difficulty in difficulties:
        baseline_metrics = baseline_by_difficulty.get(difficulty, {})
        structured_metrics = structured_by_difficulty.get(difficulty, {})
        examples = baseline_metrics.get("num_examples", structured_metrics.get("num_examples", 0))
        rows.append(
            (
                difficulty,
                str(examples),
                f"{baseline_metrics.get('exact_match_accuracy', 0.0):.4f}",
                f"{baseline_metrics.get('symbolic_accuracy', 0.0):.4f}",
                f"{structured_metrics.get('exact_match_accuracy', 0.0):.4f}",
                f"{structured_metrics.get('symbolic_accuracy', 0.0):.4f}",
            )
        )

    widths = [max(len(row[column_index]) for row in rows) for column_index in range(len(rows[0]))]

    def format_row(row: tuple[str, str, str, str, str, str]) -> str:
        return " | ".join(cell.ljust(width) for cell, width in zip(row, widths))

    separator = "-+-".join("-" * width for width in widths)
    return "\n".join([format_row(rows[0]), separator, *(format_row(row) for row in rows[1:])])


def build_text_summary(
    args: argparse.Namespace,
    baseline_metrics: dict[str, Any],
    structured_metrics: dict[str, Any],
    table: str,
    difficulty_table: str | None = None,
    structured_checkpoint_verification: dict[str, Any] | None = None,
) -> str:
    lines = [
        "Baseline vs Structured Comparison",
        f"Baseline checkpoint: {args.baseline_checkpoint}",
        f"Structured checkpoint: {args.structured_checkpoint}",
        f"Evaluation set: {args.test_path}",
        f"Per-sample output: {args.per_sample_output_path}",
        f"Examples: {baseline_metrics['num_examples']}",
        "",
        table,
        "",
    ]

    if difficulty_table is not None:
        lines.extend(
            [
                "Difficulty breakdown:",
                difficulty_table,
                "",
            ]
        )

    if structured_checkpoint_verification is not None:
        lines.extend(
            [
                "Structured checkpoint verification:",
                f"- structured_state.pt exists: {structured_checkpoint_verification['structured_state_exists']}",
                f"- node_type_embedding present: {structured_checkpoint_verification['has_node_type_embedding']}",
                f"- node_type_embedding shape: {structured_checkpoint_verification['node_type_embedding_shape']}",
                f"- node_type_embedding abs sum: {structured_checkpoint_verification['node_type_embedding_abs_sum']}",
                f"- node_type_embedding nonzero: {structured_checkpoint_verification['node_type_embedding_nonzero']}",
                "",
            ]
        )

    lines.extend(
        [
        "Sample baseline mistakes:",
        ]
    )

    if baseline_metrics["sample_mistakes"]:
        for item in baseline_metrics["sample_mistakes"]:
            lines.extend(
                [
                    f"- Prompt: {item['prompt']}",
                    f"  Target: {item['target']}",
                    f"  Raw prediction: {item['raw_prediction']}",
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
                    f"  Raw prediction: {item['raw_prediction']}",
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
    metadata_records = load_metadata_records(args.metadata_path, args.max_samples)
    if metadata_records is not None:
        if len(metadata_records) != len(dataset):
            raise ValueError(
                f"Metadata record count ({len(metadata_records)}) does not match evaluation examples ({len(dataset)})"
            )
        print(f"Loaded {len(metadata_records)} metadata records from {args.metadata_path}")

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
                stop_on_valid_answer=not args.no_stop_on_valid_answer,
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
                stop_on_valid_answer=not args.no_stop_on_valid_answer,
            )
        )

    if len(baseline_predictions) != len(dataset) or len(structured_predictions) != len(dataset):
        raise RuntimeError(
            "Comparison produced an unexpected number of predictions: "
            f"dataset={len(dataset)}, baseline={len(baseline_predictions)}, "
            f"structured={len(structured_predictions)}"
        )

    per_sample_records = build_per_sample_records(
        dataset,
        baseline_predictions,
        structured_predictions,
        metadata_records=metadata_records,
    )
    baseline_metrics = compute_accuracy_metrics(per_sample_records, "baseline")
    structured_metrics = compute_accuracy_metrics(per_sample_records, "structured")
    baseline_by_difficulty = compute_grouped_metrics(per_sample_records, "baseline", "difficulty")
    structured_by_difficulty = compute_grouped_metrics(per_sample_records, "structured", "difficulty")
    table = build_comparison_table(baseline_metrics, structured_metrics)
    difficulty_table = build_difficulty_table(baseline_by_difficulty, structured_by_difficulty)
    structured_checkpoint_verification = verify_structured_checkpoint(args.structured_checkpoint)

    print_metrics("Baseline", baseline_metrics)
    print_metrics("Structured", structured_metrics)
    print("")
    print(table)
    print("")
    print("Difficulty breakdown")
    print(difficulty_table)

    comparison = {
        "baseline_checkpoint": str(args.baseline_checkpoint),
        "structured_checkpoint": str(args.structured_checkpoint),
        "test_path": str(args.test_path),
        "metadata_path": str(args.metadata_path) if args.metadata_path is not None else None,
        "per_sample_output_path": str(args.per_sample_output_path),
        "max_samples": args.max_samples,
        "batch_size": args.batch_size,
        "max_new_tokens": args.max_new_tokens,
        "stop_on_valid_answer": not args.no_stop_on_valid_answer,
        "structured_checkpoint_verification": structured_checkpoint_verification,
        "baseline": baseline_metrics,
        "structured": structured_metrics,
        "difficulty_breakdown": {
            "baseline": baseline_by_difficulty,
            "structured": structured_by_difficulty,
        },
        "accuracy_delta": {
            "exact_match_accuracy": structured_metrics["exact_match_accuracy"]
            - baseline_metrics["exact_match_accuracy"],
            "symbolic_accuracy": structured_metrics["symbolic_accuracy"]
            - baseline_metrics["symbolic_accuracy"],
            "raw_symbolic_accuracy": structured_metrics["raw_symbolic_accuracy"]
            - baseline_metrics["raw_symbolic_accuracy"],
            "prefix_symbolic_accuracy": structured_metrics["prefix_symbolic_accuracy"]
            - baseline_metrics["prefix_symbolic_accuracy"],
        },
    }

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_path.write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    write_jsonl(args.per_sample_output_path, per_sample_records)
    args.text_output_path.parent.mkdir(parents=True, exist_ok=True)
    args.text_output_path.write_text(
        build_text_summary(
            args,
            baseline_metrics,
            structured_metrics,
            table,
            difficulty_table=difficulty_table,
            structured_checkpoint_verification=structured_checkpoint_verification,
        ),
        encoding="utf-8",
    )
    print(f"Saved comparison to {args.output_path}")
    print(f"Saved per-sample comparison to {args.per_sample_output_path}")
    print(f"Saved text comparison to {args.text_output_path}")


if __name__ == "__main__":
    main()
