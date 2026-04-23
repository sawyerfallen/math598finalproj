"""Training script for the structured Pythia-70M node-type baseline."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoTokenizer

try:
    from .node_types import ID_TO_NODE_TYPE
    from .structured_dataset import StructuredCollator, StructuredJsonlDataset
    from .structured_model import StructuredPythia
except ImportError:
    from node_types import ID_TO_NODE_TYPE
    from structured_dataset import StructuredCollator, StructuredJsonlDataset
    from structured_model import StructuredPythia


DEFAULT_MODEL_NAME = "EleutherAI/pythia-70m-deduped"
DEFAULT_EXPERIMENT_NAME = "pythia70m-structured-node-types"


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    """Append one metrics record so long runs can be inspected incrementally."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")


def write_summary(path: Path, lines: list[str]) -> None:
    """Write a short human-readable summary for the completed run."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def count_parameters(model: torch.nn.Module) -> tuple[int, int]:
    """Return total parameters and the subset that will receive gradients."""

    total_params = sum(parameter.numel() for parameter in model.parameters())
    trainable_params = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return total_params, trainable_params


def move_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    """Move all tensor values in a batch dict onto the selected device."""

    moved: dict[str, Any] = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


@torch.no_grad()
def evaluate_loss(model: StructuredPythia, dataloader: DataLoader, device: torch.device) -> float:
    """Compute mean validation/test loss over a dataloader."""

    model.eval()
    total_loss = 0.0
    total_examples = 0

    for batch in dataloader:
        batch = move_batch_to_device(batch, device)
        outputs = model(
            input_ids=batch["input_ids"],
            node_type_ids=batch["node_type_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )
        # Weight each loss by batch size so the final average is example-weighted.
        batch_size = batch["input_ids"].shape[0]
        total_loss += outputs.loss.item() * batch_size
        total_examples += batch_size

    return total_loss / max(total_examples, 1)


def maybe_save_checkpoint(
    model: StructuredPythia,
    tokenizer: Any,
    checkpoint_dir: Path,
    step_name: str,
) -> Path:
    """Save the structured model wrapper plus tokenizer into one checkpoint folder."""

    save_dir = checkpoint_dir / step_name
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    return save_dir


def parse_args() -> argparse.Namespace:
    """Define CLI flags for training, evaluation, and smoke testing."""

    parser = argparse.ArgumentParser(description="Train a structured Pythia-70M baseline with node-type embeddings.")
    parser.add_argument("--train-path", type=Path, default=Path("data/train.jsonl"))
    parser.add_argument("--val-path", type=Path, default=Path("data/val.jsonl"))
    parser.add_argument("--test-path", type=Path, default=Path("data/test.jsonl"))
    parser.add_argument("--model-name", type=str, default=DEFAULT_MODEL_NAME)
    parser.add_argument("--experiment-name", type=str, default=DEFAULT_EXPERIMENT_NAME)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--metrics-file", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--max-test-samples", type=int, default=None)
    parser.add_argument("--eval-every-steps", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args()


def log_sample_example(sample_item: dict[str, Any]) -> None:
    """Print one processed example so alignment and masking are easy to inspect."""

    prompt_len = int((sample_item["labels"] == -100).sum().item())
    prompt_node_type_names = [
        ID_TO_NODE_TYPE[int(node_type_id)]
        for node_type_id in sample_item["node_type_ids"][:prompt_len].tolist()
    ]

    print("Structured preprocessing example:")
    print(f"Prompt: {sample_item['prompt']}")
    print(f"Output: {sample_item['output']}")
    print(f"Input ids length: {sample_item['input_ids'].shape[0]}")
    print(f"Masked prompt positions: {prompt_len}")
    print(f"Node type ids length: {sample_item['node_type_ids'].shape[0]}")
    print(f"Prompt symbolic tokens: {sample_item['prompt_symbolic_tokens']}")
    print(f"Prompt symbolic node types: {sample_item['prompt_node_type_names']}")
    print(f"Aligned prompt node types: {prompt_node_type_names}")


def run_smoke_test(args: argparse.Namespace) -> None:
    """Run one structured batch through the model and print tensor shapes."""

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = StructuredJsonlDataset(args.train_path, tokenizer, max_length=args.max_length).take(
        args.max_train_samples or 2
    )
    sample_item = dataset[0]
    log_sample_example(sample_item)

    # These assertions check the exact invariants the structured trainer depends on.
    prompt_len = int((sample_item["labels"] == -100).sum().item())
    assert sample_item["node_type_ids"].shape == sample_item["input_ids"].shape
    assert torch.all(sample_item["labels"][:prompt_len] == -100)
    assert torch.any(sample_item["labels"][prompt_len:] != -100)

    collator = StructuredCollator(tokenizer)
    loader = DataLoader(dataset, batch_size=min(args.batch_size, len(dataset)), shuffle=False, collate_fn=collator)
    batch = next(iter(loader))
    assert batch["node_type_ids"].shape == batch["input_ids"].shape

    print(f"Batch input_ids shape: {tuple(batch['input_ids'].shape)}")
    print(f"Batch attention_mask shape: {tuple(batch['attention_mask'].shape)}")
    print(f"Batch labels shape: {tuple(batch['labels'].shape)}")
    print(f"Batch node_type_ids shape: {tuple(batch['node_type_ids'].shape)}")

    model = StructuredPythia(args.model_name, freeze_base=True).to(device)
    batch = move_batch_to_device(batch, device)

    with torch.no_grad():
        outputs = model(
            input_ids=batch["input_ids"],
            node_type_ids=batch["node_type_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )

    # Logits should line up with [batch, sequence, vocab].
    print(f"Logits shape: {tuple(outputs.logits.shape)}")
    print(f"Loss: {outputs.loss.item():.4f}")


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    if args.smoke_test:
        # Exit early when we only want to verify preprocessing and one forward pass.
        run_smoke_test(args)
        return

    if args.output_dir is None:
        # Keep structured experiments in their own artifact folder by default.
        args.output_dir = Path("artifacts") / "experiments" / args.experiment_name

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        # Decoder-only tokenizers often lack a pad token; eos is the standard fallback.
        tokenizer.pad_token = tokenizer.eos_token

    # The base LM is frozen; only the added node-type embedding table is trainable.
    model = StructuredPythia(args.model_name, freeze_base=True).to(device)
    total_params, trainable_params = count_parameters(model)

    train_dataset = StructuredJsonlDataset(args.train_path, tokenizer, max_length=args.max_length).take(
        args.max_train_samples
    )
    val_dataset = StructuredJsonlDataset(args.val_path, tokenizer, max_length=args.max_length).take(
        args.max_val_samples
    )
    test_dataset = StructuredJsonlDataset(args.test_path, tokenizer, max_length=args.max_length).take(
        args.max_test_samples
    )

    collator = StructuredCollator(tokenizer)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=args.num_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=args.num_workers,
    )

    optimizer = AdamW(
        # Restrict optimization to trainable parameters so the frozen base model stays untouched.
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_file = args.metrics_file or (args.output_dir / "metrics.jsonl")
    summary_file = args.output_dir / "summary.txt"
    if metrics_file.exists():
        metrics_file.unlink()

    best_val_loss = math.inf
    best_checkpoint: Path | None = None
    global_step = 0
    last_val_loss = math.inf

    sample_item = train_dataset[0]
    log_sample_example(sample_item)
    append_jsonl(
        metrics_file,
        {
            "event": "run_start",
            "model_name": args.model_name,
            "device": str(device),
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "frozen_parameters": total_params - trainable_params,
            "train_examples": len(train_dataset),
            "val_examples": len(val_dataset),
            "test_examples": len(test_dataset),
            "batch_size": args.batch_size,
            "eval_batch_size": args.eval_batch_size,
            "epochs": args.epochs,
            "learning_rate": args.lr,
            "weight_decay": args.weight_decay,
            "max_grad_norm": args.max_grad_norm,
            "max_length": args.max_length,
            "seed": args.seed,
            "sample_prompt": sample_item["prompt"],
            "sample_output": sample_item["output"],
            "sample_input_length": int(sample_item["input_ids"].shape[0]),
            "sample_masked_positions": int((sample_item["labels"] == -100).sum().item()),
        },
    )

    for epoch in range(args.epochs):
        model.train()
        progress = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}")
        epoch_loss_sum = 0.0
        epoch_steps = 0

        for batch in progress:
            batch = move_batch_to_device(batch, device)
            outputs = model(
                input_ids=batch["input_ids"],
                node_type_ids=batch["node_type_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
            )
            loss = outputs.loss
            loss.backward()
            # Clip gradients on the small trainable head for stability.
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            global_step += 1
            epoch_loss_sum += loss.item()
            epoch_steps += 1
            progress.set_postfix(loss=f"{loss.item():.4f}")

            if args.eval_every_steps > 0 and global_step % args.eval_every_steps == 0:
                # Optional mid-epoch validation follows the same loss computation as epoch-end eval.
                val_loss = evaluate_loss(model, val_loader, device)
                print(f"Step {global_step}: val_loss={val_loss:.4f}")
                append_jsonl(
                    metrics_file,
                    {
                        "event": "step_eval",
                        "epoch": epoch + 1,
                        "global_step": global_step,
                        "val_loss": val_loss,
                    },
                )
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_checkpoint = maybe_save_checkpoint(
                        model,
                        tokenizer,
                        args.output_dir,
                        f"best-step-{global_step}",
                    )

        val_loss = evaluate_loss(model, val_loader, device)
        last_val_loss = val_loss
        print(
            f"Epoch {epoch + 1}: "
            f"train_loss={epoch_loss_sum / max(epoch_steps, 1):.4f} "
            f"val_loss={val_loss:.4f}"
        )
        append_jsonl(
            metrics_file,
            {
                "event": "epoch_end",
                "epoch": epoch + 1,
                "global_step": global_step,
                "train_loss": epoch_loss_sum / max(epoch_steps, 1),
                "val_loss": val_loss,
            },
        )

        if val_loss < best_val_loss:
            # Keep the lowest-validation-loss checkpoint for final test evaluation.
            best_val_loss = val_loss
            best_checkpoint = maybe_save_checkpoint(
                model,
                tokenizer,
                args.output_dir,
                f"best-epoch-{epoch + 1}",
            )

    if best_checkpoint is not None:
        print(f"Loading best checkpoint from {best_checkpoint}")
        # Reload the best checkpoint before touching the held-out test set.
        model = StructuredPythia.from_checkpoint(best_checkpoint, freeze_base=True).to(device)
    else:
        best_checkpoint = maybe_save_checkpoint(model, tokenizer, args.output_dir, "final-model")

    # Test evaluation is run once after model-selection decisions are finished.
    test_loss = evaluate_loss(model, test_loader, device)
    print(f"Test loss: {test_loss:.4f}")
    print(f"Best checkpoint: {best_checkpoint}")
    append_jsonl(
        metrics_file,
        {
            "event": "test_end",
            "best_checkpoint": str(best_checkpoint),
            "best_val_loss": best_val_loss,
            "test_loss": test_loss,
        },
    )

    # Generation-based evaluation is intentionally skipped in v1 because
    # inputs_embeds-based generation needs extra generation plumbing to keep
    # node_type_ids aligned across decoding steps.
    write_summary(
        summary_file,
        [
            f"Experiment: {args.experiment_name}",
            f"Model: {args.model_name}",
            f"Device: {device}",
            f"Output directory: {args.output_dir}",
            f"Metrics file: {metrics_file}",
            f"Best checkpoint: {best_checkpoint}",
            f"Total parameters: {total_params}",
            f"Trainable parameters: {trainable_params}",
            f"Frozen parameters: {total_params - trainable_params}",
            f"Train examples: {len(train_dataset)}",
            f"Validation examples: {len(val_dataset)}",
            f"Test examples: {len(test_dataset)}",
            f"Epochs: {args.epochs}",
            f"Batch size: {args.batch_size}",
            f"Eval batch size: {args.eval_batch_size}",
            f"Learning rate: {args.lr}",
            f"Weight decay: {args.weight_decay}",
            f"Max grad norm: {args.max_grad_norm}",
            f"Max length: {args.max_length}",
            f"Seed: {args.seed}",
            f"Best validation loss: {best_val_loss:.6f}",
            f"Last validation loss: {last_val_loss:.6f}",
            f"Test loss: {test_loss:.6f}",
            "Generation evaluation: skipped in v1 for inputs_embeds-based structured model.",
        ],
    )
    print(f"Metrics file: {metrics_file}")
    print(f"Summary file: {summary_file}")


if __name__ == "__main__":
    main()
