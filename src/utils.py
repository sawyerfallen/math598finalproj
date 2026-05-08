"""Small shared helpers used by training and evaluation scripts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch


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


def ensure_padding_token(tokenizer: Any) -> bool:
    """Ensure decoder-only tokenizers can batch examples; return True if vocab grew."""

    if tokenizer.pad_token_id is not None:
        return False
    if tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        return False
    tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
    return True


def move_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    """Move tensor values in a mixed metadata/tensor batch to the selected device."""

    return {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }
