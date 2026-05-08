"""Shared decoding helpers for algebra-only generation."""

from __future__ import annotations

import re
from typing import Any

import torch

from .node_types import NODE_TYPE_TO_ID
from .structured_dataset import align_prompt_node_type_ids
from .structured_model import StructuredCausalLM


OTHER_NODE_TYPE_ID = NODE_TYPE_TO_ID["OTHER"]
ALLOWED_OUTPUT_CHARACTERS = set(" xyzXYZor0123456789+-*/=()^")
SOLVE_ANSWER_RE = re.compile(r"\b([xyzXYZ])\s*=\s*([+-]?\d+)\b")
SOLVE_TWO_ROOTS_RE = re.compile(
    r"\b([xyzXYZ])\s*=\s*([+-]?\d+)\s+or\s+\1\s*=\s*([+-]?\d+)\b"
)


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


def task_name_from_prompt(prompt: str) -> str:
    """Use the first prompt word as the task name."""

    return prompt.split(maxsplit=1)[0].strip().lower() if prompt.strip() else ""


def prompt_looks_quadratic(prompt: str) -> bool:
    """Detect generated quadratic solve prompts from the visible equation text."""

    return "x**2" in prompt or "x^2" in prompt


def extract_first_answer_span(prompt: str, text: str) -> str:
    """Extract the first complete answer-like span without using the target.

    This prevents a correct solve answer such as `x = 2` from being marked wrong
    solely because the model continues with extra algebraic junk afterward.
    """

    cleaned = postprocess_prediction(text)
    task = task_name_from_prompt(prompt)

    if task != "solve":
        return cleaned

    two_root_match = SOLVE_TWO_ROOTS_RE.search(cleaned)
    if two_root_match:
        variable = two_root_match.group(1).lower()
        roots = sorted({int(two_root_match.group(2)), int(two_root_match.group(3))})
        return " or ".join(f"{variable} = {root}" for root in roots)

    match = SOLVE_ANSWER_RE.search(cleaned)
    if match:
        variable = match.group(1).lower()
        value = int(match.group(2))
        return f"{variable} = {value}"
    return cleaned


def has_complete_answer_span(prompt: str, text: str) -> bool:
    """Tell the decoder when a task-specific complete answer has appeared."""

    task = task_name_from_prompt(prompt)
    if task != "solve":
        return False
    if prompt_looks_quadratic(prompt):
        return SOLVE_TWO_ROOTS_RE.search(text) is not None
    return SOLVE_ANSWER_RE.search(text) is not None


def _prompt_position_ids(attention_mask: torch.Tensor) -> torch.Tensor:
    """Position real prompt tokens from zero even when the batch is left-padded."""

    position_ids = attention_mask.long().cumsum(dim=-1) - 1
    return position_ids.masked_fill(attention_mask == 0, 0)


def _next_token_position_ids(attention_mask: torch.Tensor) -> torch.Tensor:
    """Position the current generated token correctly when using a KV cache."""

    position_ids = attention_mask.long().sum(dim=-1, keepdim=True) - 1
    return position_ids.clamp_min(0)


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
    stop_on_valid_answer: bool = True,
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
        position_ids = _prompt_position_ids(encoded["attention_mask"])
        past_key_values = None

        for _ in range(max_new_tokens):
            outputs = model(
                input_ids=next_input_ids,
                attention_mask=encoded["attention_mask"],
                position_ids=position_ids,
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

            active_rows = unfinished.clone()
            for row_index, token_id in enumerate(next_token.tolist()):
                if unfinished[row_index] and token_id != pad_token_id:
                    generated_token_rows[row_index].append(token_id)

            if stop_on_valid_answer:
                for row_index, token_ids in enumerate(generated_token_rows):
                    if not unfinished[row_index]:
                        continue
                    decoded = tokenizer.decode(token_ids, skip_special_tokens=True)
                    if has_complete_answer_span(prompts[row_index], decoded):
                        unfinished[row_index] = False

            encoded["input_ids"] = torch.cat([encoded["input_ids"], next_token.unsqueeze(1)], dim=1)
            encoded["attention_mask"] = torch.cat(
                [encoded["attention_mask"], active_rows.long().unsqueeze(1)],
                dim=1,
            )
            next_input_ids = next_token.unsqueeze(1)
            position_ids = _next_token_position_ids(encoded["attention_mask"])

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
    stop_on_valid_answer: bool = True,
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
    position_ids = _prompt_position_ids(batch["attention_mask"])
    past_key_values = None

    for _ in range(max_new_tokens):
        outputs = model(
            input_ids=next_input_ids,
            node_type_ids=next_node_type_ids,
            attention_mask=batch["attention_mask"],
            position_ids=position_ids,
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

        active_rows = unfinished.clone()
        for row_index, token_id in enumerate(next_token.tolist()):
            if unfinished[row_index] and token_id != pad_token_id:
                generated_token_rows[row_index].append(token_id)

        if stop_on_valid_answer:
            for row_index, token_ids in enumerate(generated_token_rows):
                if not unfinished[row_index]:
                    continue
                decoded = tokenizer.decode(token_ids, skip_special_tokens=True)
                if has_complete_answer_span(prompts[row_index], decoded):
                    unfinished[row_index] = False

        batch["attention_mask"] = torch.cat(
            [batch["attention_mask"], active_rows.long().unsqueeze(1)],
            dim=1,
        )
        next_input_ids = next_token.unsqueeze(1)
        next_node_type_ids = torch.full((len(prompts), 1), OTHER_NODE_TYPE_ID, dtype=torch.long, device=device)
        position_ids = _next_token_position_ids(batch["attention_mask"])

        if eos_token_id is not None:
            unfinished = unfinished & (next_token != eos_token_id)
            if not unfinished.any():
                break

    return _finalize_predictions(tokenizer, generated_token_rows)
