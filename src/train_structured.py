"""Training script for the structured causal-LM node-type baseline."""

from __future__ import annotations

import argparse
import math
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from .node_types import ID_TO_NODE_TYPE, NODE_TYPE_TO_ID
from .structured_dataset import StructuredCollator, StructuredJsonlDataset
from .structured_model import StructuredCausalLM
from .utils import (
    append_jsonl,
    count_parameters,
    ensure_padding_token,
    move_batch_to_device,
    write_summary,
)


DEFAULT_MODEL_NAME = "gpt2"
DEFAULT_EXPERIMENT_NAME = "gpt2-small-structured-node-types"
DEFAULT_ARTIFACTS_ROOT = Path("artifacts") / "models_training_info"


def resize_base_embeddings_if_needed(
    model: StructuredCausalLM,
    tokenizer: Any,
    tokenizer_vocab_grew: bool,
) -> None:
    """Keep the base LM embedding table aligned with the tokenizer."""

    if tokenizer_vocab_grew or len(tokenizer) != model.base.get_input_embeddings().num_embeddings:
        model.base.resize_token_embeddings(len(tokenizer))
        if model.freeze_base:
            for parameter in model.base.parameters():
                parameter.requires_grad = False


@torch.no_grad()
def evaluate_loss(model: StructuredCausalLM, dataloader: DataLoader, device: torch.device) -> float:
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
    model: StructuredCausalLM,
    tokenizer: Any,
    checkpoint_dir: Path,
    step_name: str,
) -> Path:
    """Save the structured model wrapper plus tokenizer into one checkpoint folder."""

    save_dir = checkpoint_dir / step_name
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    return save_dir


def summarize_named_parameters(model: torch.nn.Module, max_names: int = 20) -> dict[str, Any]:
    """Record enough parameter names to verify what is being optimized."""

    trainable = []
    frozen = []
    for name, parameter in model.named_parameters():
        record = {
            "name": name,
            "shape": list(parameter.shape),
            "numel": parameter.numel(),
        }
        if parameter.requires_grad:
            trainable.append(record)
        else:
            frozen.append(record)

    return {
        "trainable_parameter_count": len(trainable),
        "frozen_parameter_count": len(frozen),
        "trainable_parameter_names_sample": trainable[:max_names],
        "frozen_parameter_names_sample": frozen[:max_names],
        "node_type_embedding_trainable": bool(model.node_type_embedding.weight.requires_grad),
        "node_type_embedding_shape": list(model.node_type_embedding.weight.shape),
    }


def summarize_node_type_usage(dataset: StructuredJsonlDataset, max_examples: int = 128) -> dict[str, Any]:
    """Verify that structured preprocessing produces nontrivial node-type labels."""

    counts: Counter[str] = Counter()
    checked_examples = min(len(dataset), max_examples)
    prompts_with_non_other = 0

    for index in range(checked_examples):
        item = dataset[index]
        prompt_len = int((item["labels"] == -100).sum().item())
        prompt_node_type_ids = item["node_type_ids"][:prompt_len].tolist()
        prompt_node_type_names = [ID_TO_NODE_TYPE[int(node_type_id)] for node_type_id in prompt_node_type_ids]
        counts.update(prompt_node_type_names)
        if any(node_type_id != NODE_TYPE_TO_ID["OTHER"] for node_type_id in prompt_node_type_ids):
            prompts_with_non_other += 1

    total_prompt_tokens = sum(counts.values())
    non_other_tokens = total_prompt_tokens - counts.get("OTHER", 0)
    return {
        "examples_checked": checked_examples,
        "prompt_node_type_counts": dict(sorted(counts.items())),
        "unique_prompt_node_types": len(counts),
        "non_other_prompt_node_type_tokens": non_other_tokens,
        "total_prompt_node_type_tokens": total_prompt_tokens,
        "prompts_with_non_other_node_types": prompts_with_non_other,
    }


def verify_structured_checkpoint(checkpoint_dir: Path) -> dict[str, Any]:
    """Check that a saved structured checkpoint contains learned node-type weights."""

    state_path = checkpoint_dir / "structured_state.pt"
    result: dict[str, Any] = {
        "structured_state_path": str(state_path),
        "structured_state_exists": state_path.exists(),
        "has_node_type_embedding": False,
        "node_type_embedding_shape": None,
        "node_type_embedding_abs_sum": None,
        "node_type_embedding_nonzero": False,
    }
    if not state_path.exists():
        return result

    state = torch.load(state_path, map_location="cpu")
    weight = state.get("node_type_embedding", {}).get("weight")
    if weight is None:
        return result

    abs_sum = float(weight.abs().sum().item())
    result.update(
        {
            "has_node_type_embedding": True,
            "node_type_embedding_shape": list(weight.shape),
            "node_type_embedding_abs_sum": abs_sum,
            "node_type_embedding_nonzero": abs_sum > 0.0,
        }
    )
    return result


def parse_args() -> argparse.Namespace:
    """Define CLI flags for training, evaluation, and smoke testing."""

    parser = argparse.ArgumentParser(description="Train a structured causal LM with node-type embeddings.")
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
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--max-test-samples", type=int, default=None)
    parser.add_argument("--eval-every-steps", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--save-best-checkpoint",
        action="store_true",
        help="Also save and reload the lowest-validation-loss checkpoint in addition to final-model.",
    )
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument(
        "--freeze-base",
        action="store_true",
        help="Freeze the base LM and train only the added node-type embeddings.",
    )
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
    tokenizer_vocab_grew = ensure_padding_token(tokenizer)

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

    model = StructuredCausalLM(args.model_name, freeze_base=args.freeze_base).to(device)
    resize_base_embeddings_if_needed(model, tokenizer, tokenizer_vocab_grew)
    model.base.config.pad_token_id = tokenizer.pad_token_id
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
        args.output_dir = DEFAULT_ARTIFACTS_ROOT / args.experiment_name

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer_vocab_grew = ensure_padding_token(tokenizer)

    # By default, fine-tune the base LM and the added node-type embedding table together.
    model = StructuredCausalLM(args.model_name, freeze_base=args.freeze_base).to(device)
    resize_base_embeddings_if_needed(model, tokenizer, tokenizer_vocab_grew)
    model.base.config.pad_token_id = tokenizer.pad_token_id
    total_params, trainable_params = count_parameters(model)
    parameter_summary = summarize_named_parameters(model)

    train_dataset = StructuredJsonlDataset(args.train_path, tokenizer, max_length=args.max_length).take(
        args.max_train_samples
    )
    val_dataset = StructuredJsonlDataset(args.val_path, tokenizer, max_length=args.max_length).take(
        args.max_val_samples
    )
    test_dataset = StructuredJsonlDataset(args.test_path, tokenizer, max_length=args.max_length).take(
        args.max_test_samples
    )
    node_type_usage = summarize_node_type_usage(train_dataset)

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
        # If --freeze-base is used, this naturally excludes the frozen base-model weights.
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
    print("Structured verification:")
    print(f"Trainable parameter tensors: {parameter_summary['trainable_parameter_count']}")
    print(f"Frozen parameter tensors: {parameter_summary['frozen_parameter_count']}")
    print(f"Node-type embedding trainable: {parameter_summary['node_type_embedding_trainable']}")
    print(f"Node-type usage summary: {node_type_usage}")
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
            "freeze_base": args.freeze_base,
            "save_best_checkpoint": args.save_best_checkpoint,
            "sample_prompt": sample_item["prompt"],
            "sample_output": sample_item["output"],
            "sample_input_length": int(sample_item["input_ids"].shape[0]),
            "sample_masked_positions": int((sample_item["labels"] == -100).sum().item()),
            "parameter_summary": parameter_summary,
            "node_type_usage": node_type_usage,
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
            # Clip gradients before the optimizer step, whether the base LM is frozen or fully fine-tuned.
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            global_step += 1
            epoch_loss_sum += loss.item()
            epoch_steps += 1
            progress.set_postfix(loss=f"{loss.item():.4f}")
            append_jsonl(
                metrics_file,
                {
                    "event": "batch_end",
                    "epoch": epoch + 1,
                    "global_step": global_step,
                    "epoch_step": epoch_steps,
                    "train_loss": loss.item(),
                },
            )

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
                if args.save_best_checkpoint and val_loss < best_val_loss:
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

        if args.save_best_checkpoint and val_loss < best_val_loss:
            # Keep the lowest-validation-loss checkpoint for final test evaluation.
            best_val_loss = val_loss
            best_checkpoint = maybe_save_checkpoint(
                model,
                tokenizer,
                args.output_dir,
                f"best-epoch-{epoch + 1}",
            )

    final_checkpoint = maybe_save_checkpoint(model, tokenizer, args.output_dir, "final-model")
    final_checkpoint_verification = verify_structured_checkpoint(final_checkpoint)

    if args.save_best_checkpoint and best_checkpoint is not None:
        print(f"Loading best checkpoint from {best_checkpoint}")
        # Reload the best checkpoint before touching the held-out test set.
        model = StructuredCausalLM.from_checkpoint(best_checkpoint, freeze_base=args.freeze_base).to(device)
        model.base.config.pad_token_id = tokenizer.pad_token_id
    else:
        best_checkpoint = final_checkpoint
    best_checkpoint_verification = verify_structured_checkpoint(best_checkpoint)

    # Test evaluation is run once after model-selection decisions are finished.
    test_loss = evaluate_loss(model, test_loader, device)
    print(f"Test loss: {test_loss:.4f}")
    print(f"Best checkpoint: {best_checkpoint}")
    append_jsonl(
        metrics_file,
        {
            "event": "test_end",
            "best_checkpoint": str(best_checkpoint),
            "final_checkpoint": str(final_checkpoint),
            "best_val_loss": best_val_loss,
            "test_loss": test_loss,
            "best_checkpoint_verification": best_checkpoint_verification,
            "final_checkpoint_verification": final_checkpoint_verification,
        },
    )

    write_summary(
        summary_file,
        [
            f"Experiment: {args.experiment_name}",
            f"Model: {args.model_name}",
            f"Device: {device}",
            f"Output directory: {args.output_dir}",
            f"Metrics file: {metrics_file}",
            f"Best checkpoint: {best_checkpoint}",
            f"Final checkpoint: {final_checkpoint}",
            f"Total parameters: {total_params}",
            f"Trainable parameters: {trainable_params}",
            f"Frozen parameters: {total_params - trainable_params}",
            f"Trainable parameter tensors: {parameter_summary['trainable_parameter_count']}",
            f"Frozen parameter tensors: {parameter_summary['frozen_parameter_count']}",
            "Trainable parameter sample: "
            + ", ".join(item["name"] for item in parameter_summary["trainable_parameter_names_sample"][:10]),
            f"Freeze base: {args.freeze_base}",
            f"Node-type embedding trainable: {parameter_summary['node_type_embedding_trainable']}",
            f"Node-type embedding shape: {parameter_summary['node_type_embedding_shape']}",
            f"Node-type usage over {node_type_usage['examples_checked']} train examples: {node_type_usage['prompt_node_type_counts']}",
            f"Non-OTHER prompt node-type tokens: {node_type_usage['non_other_prompt_node_type_tokens']} / {node_type_usage['total_prompt_node_type_tokens']}",
            f"Prompts with non-OTHER node types: {node_type_usage['prompts_with_non_other_node_types']}",
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
            f"Best checkpoint structured_state.pt exists: {best_checkpoint_verification['structured_state_exists']}",
            f"Best checkpoint node_type_embedding present: {best_checkpoint_verification['has_node_type_embedding']}",
            f"Best checkpoint node_type_embedding shape: {best_checkpoint_verification['node_type_embedding_shape']}",
            f"Best checkpoint node_type_embedding abs sum: {best_checkpoint_verification['node_type_embedding_abs_sum']}",
            f"Best checkpoint node_type_embedding nonzero: {best_checkpoint_verification['node_type_embedding_nonzero']}",
            f"Final checkpoint structured_state.pt exists: {final_checkpoint_verification['structured_state_exists']}",
            f"Final checkpoint node_type_embedding nonzero: {final_checkpoint_verification['node_type_embedding_nonzero']}",
            f"Save best checkpoint: {args.save_best_checkpoint}",
            "Generation evaluation: run separately with compare-models on saved checkpoints.",
        ],
    )
    print(f"Metrics file: {metrics_file}")
    print(f"Summary file: {summary_file}")
    print(f"Final checkpoint: {final_checkpoint}")


if __name__ == "__main__":
    main()
