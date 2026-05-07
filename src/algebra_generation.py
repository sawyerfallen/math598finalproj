"""Shared decoding helpers for algebra-only generation."""

from __future__ import annotations

from typing import Any

import torch

try:
    from .node_types import NODE_TYPE_TO_ID
    from .structured_dataset import align_prompt_node_type_ids
    from .structured_model import StructuredCausalLM
except ImportError:
    from node_types import NODE_TYPE_TO_ID
    from structured_dataset import align_prompt_node_type_ids
    from structured_model import StructuredCausalLM


OTHER_NODE_TYPE_ID = NODE_TYPE_TO_ID["OTHER"]
ALLOWED_OUTPUT_CHARACTERS = set(" xyzXYZ0123456789+-*/=()^")


def _decoded_token_text(tokenizer: Any, token_id: int) -> str:
    return tokenizer.decode([token_id], skip_special_tokens=False, clean_up_tokenization_spaces=False)


def build_allowed_token_mask(
    tokenizer: Any,
    device: torch.device,
    vocab_size: int | None = None,
) -> torch.Tensor:
    """Allow only tokens that decode to simple algebra text plus EOS."""

    if vocab_size is None:
        vocab_size = len(tokenizer)
    allowed_mask = torch.zeros(vocab_size, dtype=torch.bool)

    for token_id in range(vocab_size):
        if token_id == tokenizer.eos_token_id:
            allowed_mask[token_id] = True
            continue

        decoded = _decoded_token_text(tokenizer, token_id)
        if not decoded:
            continue
        if all(character in ALLOWED_OUTPUT_CHARACTERS for character in decoded):
            allowed_mask[token_id] = True

    if tokenizer.pad_token_id is not None:
        allowed_mask[tokenizer.pad_token_id] = False

    return allowed_mask.to(device)


def postprocess_prediction(text: str) -> str:
    """Trim whitespace and cut off anything after the first newline if one appears."""

    return text.splitlines()[0].strip()


def _apply_output_constraints(logits: torch.Tensor, allowed_token_mask: torch.Tensor) -> torch.Tensor:
    constrained = logits.clone()
    constrained[:, ~allowed_token_mask] = -torch.inf
    return constrained


def _finalize_predictions(
    tokenizer: Any,
    generated_token_rows: list[list[int]],
) -> list[str]:
    predictions = []
    for token_ids in generated_token_rows:
        decoded = tokenizer.decode(token_ids, skip_special_tokens=True)
        predictions.append(postprocess_prediction(decoded))
    return predictions


@torch.no_grad()
def generate_baseline_predictions(
    model: Any,
    tokenizer: Any,
    prompts: list[str],
    device: torch.device,
    max_new_tokens: int,
    allowed_token_mask: torch.Tensor,
) -> list[str]:
    """Greedy decode baseline answers with algebra-only token constraints."""

    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"

    try:
        encoded = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            add_special_tokens=False,
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}

        generated_token_rows: list[list[int]] = [[] for _ in prompts]
        unfinished = torch.ones(encoded["input_ids"].shape[0], dtype=torch.bool, device=device)
        eos_token_id = tokenizer.eos_token_id
        pad_token_id = tokenizer.pad_token_id or 0
        next_input_ids = encoded["input_ids"]
        past_key_values = None

        for _ in range(max_new_tokens):
            outputs = model(
                input_ids=next_input_ids,
                attention_mask=encoded["attention_mask"],
                past_key_values=past_key_values,
                use_cache=True,
            )
            past_key_values = outputs.past_key_values
            next_token_logits = _apply_output_constraints(outputs.logits[:, -1, :], allowed_token_mask)
            next_token = next_token_logits.argmax(dim=-1)

            if eos_token_id is not None:
                next_token = torch.where(
                    unfinished,
                    next_token,
                    torch.full_like(next_token, pad_token_id),
                )

            for row_index, token_id in enumerate(next_token.tolist()):
                if unfinished[row_index] and token_id != pad_token_id:
                    generated_token_rows[row_index].append(token_id)

            encoded["input_ids"] = torch.cat([encoded["input_ids"], next_token.unsqueeze(1)], dim=1)
            encoded["attention_mask"] = torch.cat(
                [encoded["attention_mask"], unfinished.long().unsqueeze(1)],
                dim=1,
            )
            next_input_ids = next_token.unsqueeze(1)

            if eos_token_id is not None:
                unfinished = unfinished & (next_token != eos_token_id)
                if not unfinished.any():
                    break

        return _finalize_predictions(tokenizer, generated_token_rows)
    finally:
        tokenizer.padding_side = original_padding_side


def build_structured_prompt_batch(prompts: list[str], tokenizer: Any) -> dict[str, torch.Tensor]:
    """Create left-padded prompt tensors and aligned node-type ids for structured decoding."""

    input_id_rows: list[list[int]] = []
    node_type_rows: list[list[int]] = []

    for prompt in prompts:
        input_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        node_type_ids, _, _ = align_prompt_node_type_ids(prompt, tokenizer)
        if len(input_ids) != len(node_type_ids):
            raise ValueError(f"Structured prompt alignment length mismatch for prompt: {prompt!r}")

        input_id_rows.append(input_ids)
        node_type_rows.append(node_type_ids)

    max_len = max(len(row) for row in input_id_rows)
    padded_input_ids = []
    padded_attention_masks = []
    padded_node_type_ids = []

    for input_ids, node_type_ids in zip(input_id_rows, node_type_rows):
        pad_len = max_len - len(input_ids)
        padded_input_ids.append([tokenizer.pad_token_id] * pad_len + input_ids)
        padded_attention_masks.append([0] * pad_len + [1] * len(input_ids))
        padded_node_type_ids.append([OTHER_NODE_TYPE_ID] * pad_len + node_type_ids)

    return {
        "input_ids": torch.tensor(padded_input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(padded_attention_masks, dtype=torch.long),
        "node_type_ids": torch.tensor(padded_node_type_ids, dtype=torch.long),
    }


@torch.no_grad()
def generate_structured_predictions(
    model: StructuredCausalLM,
    tokenizer: Any,
    prompts: list[str],
    device: torch.device,
    max_new_tokens: int,
    allowed_token_mask: torch.Tensor,
) -> list[str]:
    """Greedy decode structured answers while appending OTHER node types for new tokens."""

    batch = build_structured_prompt_batch(prompts, tokenizer)
    batch = {key: value.to(device) for key, value in batch.items()}

    generated_token_rows: list[list[int]] = [[] for _ in prompts]
    unfinished = torch.ones(batch["input_ids"].shape[0], dtype=torch.bool, device=device)
    eos_token_id = tokenizer.eos_token_id
    pad_token_id = tokenizer.pad_token_id or 0
    next_input_ids = batch["input_ids"]
    next_node_type_ids = batch["node_type_ids"]
    past_key_values = None

    for _ in range(max_new_tokens):
        outputs = model(
            input_ids=next_input_ids,
            node_type_ids=next_node_type_ids,
            attention_mask=batch["attention_mask"],
            past_key_values=past_key_values,
            use_cache=True,
        )
        past_key_values = outputs.past_key_values
        next_token_logits = _apply_output_constraints(outputs.logits[:, -1, :], allowed_token_mask)
        next_token = next_token_logits.argmax(dim=-1)

        if eos_token_id is not None:
            next_token = torch.where(
                unfinished,
                next_token,
                torch.full_like(next_token, pad_token_id),
            )

        for row_index, token_id in enumerate(next_token.tolist()):
            if unfinished[row_index] and token_id != pad_token_id:
                generated_token_rows[row_index].append(token_id)

        batch["attention_mask"] = torch.cat(
            [batch["attention_mask"], unfinished.long().unsqueeze(1)],
            dim=1,
        )
        next_input_ids = next_token.unsqueeze(1)
        next_node_type_ids = torch.full((len(prompts), 1), OTHER_NODE_TYPE_ID, dtype=torch.long, device=device)

        if eos_token_id is not None:
            unfinished = unfinished & (next_token != eos_token_id)
            if not unfinished.any():
                break

    return _finalize_predictions(tokenizer, generated_token_rows)
