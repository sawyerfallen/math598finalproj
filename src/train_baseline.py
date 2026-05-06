"""Baseline GPT-2 fine-tuning for algebra prompt/output pairs."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import sympy as sp
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from .algebra_generation import (
        build_allowed_token_mask,
        generate_baseline_predictions as constrained_generate_baseline_predictions,
    )
    from .utils import (
        append_jsonl,
        count_parameters,
        ensure_padding_token,
        move_batch_to_device,
        write_summary,
    )
except ImportError:
    from algebra_generation import (
        build_allowed_token_mask,
        generate_baseline_predictions as constrained_generate_baseline_predictions,
    )
    from utils import (
        append_jsonl,
        count_parameters,
        ensure_padding_token,
        move_batch_to_device,
        write_summary,
    )


DEFAULT_MODEL_NAME = "gpt2"
DEFAULT_EXPERIMENT_NAME = "gpt2-small-baseline"
SYMPY_LOCALS = {name: sp.Symbol(name) for name in ("x", "y", "z")}


@dataclass
class AlgebraExample:
    prompt: str
    output: str


class JsonlAlgebraDataset(Dataset[AlgebraExample]):
    """Read prompt/output examples from the project JSONL format."""

    def __init__(self, path: Path):
        self.path = path
        self.examples = self._load_examples(path)

    @staticmethod
    def _load_examples(path: Path) -> list[AlgebraExample]:
        examples: list[AlgebraExample] = []
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                record = json.loads(line)
                if "prompt" not in record or "output" not in record:
                    raise ValueError(f"{path}:{line_number} is missing 'prompt' or 'output'")
                examples.append(
                    AlgebraExample(
                        prompt=str(record["prompt"]),
                        output=str(record["output"]),
                    )
                )
        return examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> AlgebraExample:
        return self.examples[index]

    def take(self, count: int | None) -> "JsonlAlgebraDataset":
        """Return a lightweight prefix subset for smoke tests and smaller runs."""

        if count is None or count >= len(self.examples):
            return self
        subset = JsonlAlgebraDataset.__new__(JsonlAlgebraDataset)
        subset.path = self.path
        subset.examples = self.examples[:count]
        return subset


class CausalLmPromptMaskingDataset(Dataset[dict[str, torch.Tensor | str]]):
    """Tokenize prompt+answer text while masking prompt tokens out of the loss."""

    def __init__(
        self,
        examples: JsonlAlgebraDataset,
        tokenizer: Any,
        max_length: int | None = None,
    ):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        example = self.examples[index]
        eos_text = self.tokenizer.eos_token or ""
        full_text = f"{example.prompt} {example.output}{eos_text}"

        full_enc = self.tokenizer(
            full_text,
            add_special_tokens=False,
            truncation=self.max_length is not None,
            max_length=self.max_length,
            return_attention_mask=True,
        )
        prompt_enc = self.tokenizer(
            example.prompt,
            add_special_tokens=False,
            truncation=self.max_length is not None,
            max_length=self.max_length,
            return_attention_mask=False,
        )

        input_ids = torch.tensor(full_enc["input_ids"], dtype=torch.long)
        attention_mask = torch.tensor(full_enc["attention_mask"], dtype=torch.long)
        labels = input_ids.clone()

        # The model sees the prompt as context, but gradients only come from answer tokens.
        prompt_len = min(len(prompt_enc["input_ids"]), labels.shape[0])
        labels[:prompt_len] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "prompt": example.prompt,
            "output": example.output,
        }


class PromptMaskingCollator:
    """Right-pad variable-length causal LM examples into one batch."""

    def __init__(self, tokenizer: Any):
        self.pad_token_id = tokenizer.pad_token_id
        if self.pad_token_id is None:
            raise ValueError("Tokenizer must have a pad_token_id before batching.")

    def __call__(self, batch: list[dict[str, torch.Tensor | str]]) -> dict[str, Any]:
        max_len = max(item["input_ids"].shape[0] for item in batch)

        input_ids = []
        attention_masks = []
        labels = []
        prompts = []
        outputs = []

        for item in batch:
            seq_len = item["input_ids"].shape[0]
            pad_len = max_len - seq_len

            input_ids.append(
                torch.cat(
                    [
                        item["input_ids"],
                        torch.full((pad_len,), self.pad_token_id, dtype=torch.long),
                    ]
                )
            )
            attention_masks.append(
                torch.cat(
                    [
                        item["attention_mask"],
                        torch.zeros((pad_len,), dtype=torch.long),
                    ]
                )
            )
            labels.append(
                torch.cat(
                    [
                        item["labels"],
                        torch.full((pad_len,), -100, dtype=torch.long),
                    ]
                )
            )
            prompts.append(item["prompt"])
            outputs.append(item["output"])

        return {
            "input_ids": torch.stack(input_ids),
            "attention_mask": torch.stack(attention_masks),
            "labels": torch.stack(labels),
            "prompts": prompts,
            "outputs": outputs,
        }


@torch.no_grad()
def evaluate_loss(model: torch.nn.Module, dataloader: DataLoader, device: torch.device) -> float:
    model.eval()
    total_loss = 0.0
    total_examples = 0

    for batch in dataloader:
        batch = move_batch_to_device(batch, device)
        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )
        batch_size = batch["input_ids"].shape[0]
        total_loss += outputs.loss.item() * batch_size
        total_examples += batch_size

    return total_loss / max(total_examples, 1)


@torch.no_grad()
def evaluate_generation_metrics(
    model: torch.nn.Module,
    tokenizer: Any,
    dataset: JsonlAlgebraDataset,
    device: torch.device,
    batch_size: int,
    max_new_tokens: int,
    allowed_token_mask: torch.Tensor,
) -> dict[str, float]:
    model.eval()
    predictions: list[str] = []

    for start in tqdm(range(0, len(dataset), batch_size), desc="Generating", leave=False):
        prompts = [example.prompt for example in dataset.examples[start:start + batch_size]]
        predictions.extend(
            constrained_generate_baseline_predictions(
                model=model,
                tokenizer=tokenizer,
                prompts=prompts,
                device=device,
                max_new_tokens=max_new_tokens,
                allowed_token_mask=allowed_token_mask,
            )
        )

    exact_matches = 0
    symbolic_matches = 0

    for prediction, example in zip(predictions, dataset.examples):
        target = example.output.strip()
        if prediction == target:
            exact_matches += 1
        if is_symbolically_equivalent(prediction, target):
            symbolic_matches += 1

    total = max(len(dataset), 1)
    return {
        "exact_match_accuracy": exact_matches / total,
        "symbolic_accuracy": symbolic_matches / total,
    }


def maybe_save_checkpoint(
    model: torch.nn.Module,
    tokenizer: Any,
    checkpoint_dir: Path,
    step_name: str,
) -> Path:
    save_dir = checkpoint_dir / step_name
    save_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    return save_dir


def parse_symbolic_answer(text: str) -> sp.Basic | sp.Equality | None:
    """Parse a generated algebra answer into a SymPy expression/equality when possible."""

    cleaned = text.strip()
    if not cleaned:
        return None

    try:
        if "=" in cleaned:
            left_text, right_text = cleaned.split("=", maxsplit=1)
            left_expr = sp.sympify(left_text.strip(), locals=SYMPY_LOCALS)
            right_expr = sp.sympify(right_text.strip(), locals=SYMPY_LOCALS)
            return sp.Eq(left_expr, right_expr, evaluate=False)
        return sp.sympify(cleaned, locals=SYMPY_LOCALS)
    except (sp.SympifyError, TypeError, ValueError, SyntaxError):
        return None


def is_symbolically_equivalent(prediction: str, target: str) -> bool:
    """Compare answers by symbolic meaning, allowing algebraically equivalent text."""

    parsed_prediction = parse_symbolic_answer(prediction)
    parsed_target = parse_symbolic_answer(target)

    if parsed_prediction is None or parsed_target is None:
        return False

    valid_prediction = isinstance(parsed_prediction, (sp.Basic, sp.Equality))
    valid_target = isinstance(parsed_target, (sp.Basic, sp.Equality))
    if not valid_prediction or not valid_target:
        return False

    if isinstance(parsed_prediction, sp.Equality) and isinstance(parsed_target, sp.Equality):
        prediction_residual = sp.simplify(parsed_prediction.lhs - parsed_prediction.rhs)
        target_residual = sp.simplify(parsed_target.lhs - parsed_target.rhs)
        return bool(
            sp.simplify(prediction_residual - target_residual) == 0
            or sp.simplify(prediction_residual + target_residual) == 0
        )

    if isinstance(parsed_prediction, sp.Equality) or isinstance(parsed_target, sp.Equality):
        return False

    try:
        return bool(sp.simplify(parsed_prediction - parsed_target) == 0)
    except TypeError:
        return False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune GPT-2 small on algebra prompt/output pairs.")
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
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--max-test-samples", type=int, default=None)
    parser.add_argument("--eval-every-steps", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    if args.output_dir is None:
        args.output_dir = Path("artifacts") / "experiments" / args.experiment_name

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # GPT-2 has no dedicated pad token, so use EOS as padding without growing the vocab.
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer_vocab_grew = ensure_padding_token(tokenizer)

    model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.float32)
    if tokenizer_vocab_grew or len(tokenizer) != model.get_input_embeddings().num_embeddings:
        model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = tokenizer.pad_token_id
    model.to(device)
    total_params, trainable_params = count_parameters(model)
    allowed_token_mask = build_allowed_token_mask(tokenizer, device, vocab_size=model.config.vocab_size)

    # Baseline examples use only text; prompt labels are masked inside the dataset wrapper.
    train_examples = JsonlAlgebraDataset(args.train_path).take(args.max_train_samples)
    val_examples = JsonlAlgebraDataset(args.val_path).take(args.max_val_samples)
    test_examples = JsonlAlgebraDataset(args.test_path).take(args.max_test_samples)

    train_dataset = CausalLmPromptMaskingDataset(train_examples, tokenizer, max_length=args.max_length)
    val_dataset = CausalLmPromptMaskingDataset(val_examples, tokenizer, max_length=args.max_length)
    test_dataset = CausalLmPromptMaskingDataset(test_examples, tokenizer, max_length=args.max_length)

    collator = PromptMaskingCollator(tokenizer)

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

    # Full fine-tuning: every GPT-2 parameter is trainable in the baseline.
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_file = args.metrics_file or (args.output_dir / "metrics.jsonl")
    summary_file = args.output_dir / "summary.txt"

    if metrics_file.exists():
        metrics_file.unlink()

    best_val_loss = math.inf
    best_checkpoint: Path | None = None
    global_step = 0
    last_val_loss = math.inf
    last_val_exact_match_accuracy = 0.0
    last_val_symbolic_accuracy = 0.0

    sample_item = train_dataset[0]
    print("Baseline preprocessing example:")
    print(f"Prompt: {sample_item['prompt']}")
    print(f"Output: {sample_item['output']}")
    print(f"Input ids length: {sample_item['input_ids'].shape[0]}")
    print(f"Masked label positions: {(sample_item['labels'] == -100).sum().item()}")
    append_jsonl(
        metrics_file,
        {
            "event": "run_start",
            "model_name": args.model_name,
            "device": str(device),
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "frozen_parameters": total_params - trainable_params,
            "train_examples": len(train_examples),
            "val_examples": len(val_examples),
            "test_examples": len(test_examples),
            "batch_size": args.batch_size,
            "eval_batch_size": args.eval_batch_size,
            "epochs": args.epochs,
            "learning_rate": args.lr,
            "weight_decay": args.weight_decay,
            "max_grad_norm": args.max_grad_norm,
            "max_length": args.max_length,
            "max_new_tokens": args.max_new_tokens,
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
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
            )
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            global_step += 1
            epoch_loss_sum += loss.item()
            epoch_steps += 1
            progress.set_postfix(loss=f"{loss.item():.4f}")

            if args.eval_every_steps > 0 and global_step % args.eval_every_steps == 0:
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
        val_metrics = evaluate_generation_metrics(
            model,
            tokenizer,
            val_examples,
            device,
            batch_size=args.eval_batch_size,
            max_new_tokens=args.max_new_tokens,
            allowed_token_mask=allowed_token_mask,
        )
        last_val_loss = val_loss
        last_val_exact_match_accuracy = val_metrics["exact_match_accuracy"]
        last_val_symbolic_accuracy = val_metrics["symbolic_accuracy"]
        print(
            f"Epoch {epoch + 1}: "
            f"train_loss={epoch_loss_sum / max(epoch_steps, 1):.4f} "
            f"val_loss={val_loss:.4f} "
            f"val_exact_match_accuracy={val_metrics['exact_match_accuracy']:.4f} "
            f"val_symbolic_accuracy={val_metrics['symbolic_accuracy']:.4f}"
        )
        append_jsonl(
            metrics_file,
            {
                "event": "epoch_end",
                "epoch": epoch + 1,
                "global_step": global_step,
                "train_loss": epoch_loss_sum / max(epoch_steps, 1),
                "val_loss": val_loss,
                "val_exact_match_accuracy": val_metrics["exact_match_accuracy"],
                "val_generation_accuracy": val_metrics["exact_match_accuracy"],
                "val_symbolic_accuracy": val_metrics["symbolic_accuracy"],
            },
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_checkpoint = maybe_save_checkpoint(
                model,
                tokenizer,
                args.output_dir,
                f"best-epoch-{epoch + 1}",
            )

    if best_checkpoint is not None:
        print(f"Loading best checkpoint from {best_checkpoint}")
        model = AutoModelForCausalLM.from_pretrained(best_checkpoint, torch_dtype=torch.float32).to(device)
        model.config.pad_token_id = tokenizer.pad_token_id
    else:
        best_checkpoint = maybe_save_checkpoint(model, tokenizer, args.output_dir, "final-model")

    test_loss = evaluate_loss(model, test_loader, device)
    test_metrics = evaluate_generation_metrics(
        model,
        tokenizer,
        test_examples,
        device,
        batch_size=args.eval_batch_size,
        max_new_tokens=args.max_new_tokens,
        allowed_token_mask=allowed_token_mask,
    )
    test_exact_match_accuracy = test_metrics["exact_match_accuracy"]
    test_symbolic_accuracy = test_metrics["symbolic_accuracy"]
    print(f"Test loss: {test_loss:.4f}")
    print(f"Test exact match accuracy: {test_exact_match_accuracy:.4f}")
    print(f"Test symbolic accuracy: {test_symbolic_accuracy:.4f}")
    print(f"Best checkpoint: {best_checkpoint}")
    append_jsonl(
        metrics_file,
        {
            "event": "test_end",
            "best_checkpoint": str(best_checkpoint),
            "best_val_loss": best_val_loss,
            "test_loss": test_loss,
            "test_exact_match_accuracy": test_exact_match_accuracy,
            "test_generation_accuracy": test_exact_match_accuracy,
            "test_symbolic_accuracy": test_symbolic_accuracy,
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
            f"Total parameters: {total_params}",
            f"Trainable parameters: {trainable_params}",
            f"Frozen parameters: {total_params - trainable_params}",
            f"Train examples: {len(train_examples)}",
            f"Validation examples: {len(val_examples)}",
            f"Test examples: {len(test_examples)}",
            f"Epochs: {args.epochs}",
            f"Batch size: {args.batch_size}",
            f"Eval batch size: {args.eval_batch_size}",
            f"Learning rate: {args.lr}",
            f"Weight decay: {args.weight_decay}",
            f"Max grad norm: {args.max_grad_norm}",
            f"Max length: {args.max_length}",
            f"Max new tokens: {args.max_new_tokens}",
            f"Seed: {args.seed}",
            f"Best validation loss: {best_val_loss:.6f}",
            f"Last validation loss: {last_val_loss:.6f}",
            f"Last validation exact match accuracy: {last_val_exact_match_accuracy:.6f}",
            f"Last validation symbolic accuracy: {last_val_symbolic_accuracy:.6f}",
            f"Test loss: {test_loss:.6f}",
            f"Test exact match accuracy: {test_exact_match_accuracy:.6f}",
            f"Test symbolic accuracy: {test_symbolic_accuracy:.6f}",
        ],
    )
    print(f"Metrics file: {metrics_file}")
    print(f"Summary file: {summary_file}")


if __name__ == "__main__":
    main()
