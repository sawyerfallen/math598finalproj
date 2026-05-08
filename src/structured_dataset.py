"""Dataset and collation utilities for structured prompt annotations."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from .node_types import ID_TO_NODE_TYPE, NODE_TYPE_TO_ID
from .parser import tokenize_prompt_with_node_types


OTHER_NODE_TYPE_ID = NODE_TYPE_TO_ID["OTHER"]


@dataclass
class StructuredAlgebraExample:
    """Raw JSONL example with only prompt/output text."""

    prompt: str
    output: str


def align_prompt_node_type_ids(prompt: str, tokenizer: Any) -> tuple[list[int], list[str], list[str]]:
    """Align symbolic prompt node types to tokenizer subpieces.

    GPT-style BPE tokenizers can merge across symbolic-token boundaries
    (for example `)*(`). We therefore project node types onto tokenizer
    offsets and mark mixed-type merged pieces as `OTHER`.
    """

    symbolic_tokens, node_type_names = tokenize_prompt_with_node_types(prompt)
    cursor = 0
    symbolic_spans: list[tuple[int, int]] = []

    for symbolic_token in symbolic_tokens:
        # Skip inter-token spaces before locating the next symbolic token in the raw prompt.
        while cursor < len(prompt) and prompt[cursor].isspace():
            cursor += 1

        if not prompt.startswith(symbolic_token, cursor):
            raise ValueError(
                f"Could not align symbolic token {symbolic_token!r} in prompt {prompt!r} at position {cursor}."
            )

        # Track character spans so we can project symbolic node types onto tokenizer offsets.
        token_start = cursor
        token_end = cursor + len(symbolic_token)
        symbolic_spans.append((token_start, token_end))
        cursor = token_end

    trailing_text = prompt[cursor:]
    if trailing_text and not trailing_text.isspace():
        raise ValueError(f"Parser did not consume the full prompt: {prompt!r}")

    prompt_enc = tokenizer(prompt, add_special_tokens=False, return_offsets_mapping=True)
    if "offset_mapping" not in prompt_enc:
        raise ValueError("Tokenizer did not return offset_mapping for structured alignment.")

    aligned_node_type_ids: list[int] = []
    for token_start, token_end in prompt_enc["offset_mapping"]:
        # Collect all symbolic node types that overlap this tokenizer piece.
        overlapping_types = []
        for (span_start, span_end), node_type_name in zip(symbolic_spans, node_type_names):
            overlap = min(token_end, span_end) - max(token_start, span_start)
            if overlap > 0:
                overlapping_types.append(node_type_name)

        if not overlapping_types:
            # This should be rare, but OTHER is the safest fallback when no symbolic token overlaps.
            aligned_node_type_ids.append(OTHER_NODE_TYPE_ID)
            continue

        unique_types = set(overlapping_types)
        if len(unique_types) == 1:
            # If the tokenizer piece covers exactly one symbolic type, keep that label.
            aligned_node_type_ids.append(NODE_TYPE_TO_ID[overlapping_types[0]])
        else:
            # Mixed merges like `)*(` do not have one clean symbolic type, so mark them as OTHER.
            aligned_node_type_ids.append(OTHER_NODE_TYPE_ID)

    return aligned_node_type_ids, symbolic_tokens, node_type_names


class StructuredJsonlDataset(Dataset[dict[str, torch.Tensor | str | list[str]]]):
    """JSONL dataset that augments prompt tokens with node-type ids."""

    def __init__(self, path: Path, tokenizer: Any, max_length: int | None = None):
        self.path = path
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = self._load_examples(path)

    @staticmethod
    def _load_examples(path: Path) -> list[StructuredAlgebraExample]:
        examples: list[StructuredAlgebraExample] = []
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                record = json.loads(line)
                if "prompt" not in record or "output" not in record:
                    raise ValueError(f"{path}:{line_number} is missing 'prompt' or 'output'")
                examples.append(
                    StructuredAlgebraExample(
                        prompt=str(record["prompt"]),
                        output=str(record["output"]),
                    )
                )
        return examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str | list[str]]:
        example = self.examples[index]
        # Append EOS so the model sees an explicit end-of-answer target during training.
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

        # Mask the prompt region so loss is only applied on the answer tokens.
        prompt_len = min(len(prompt_enc["input_ids"]), input_ids.shape[0])
        labels[:prompt_len] = -100

        prompt_node_type_ids, symbolic_tokens, node_type_names = align_prompt_node_type_ids(
            example.prompt,
            self.tokenizer,
        )
        # Only prompt tokens receive parsed node types; the answer region defaults to OTHER.
        prompt_node_type_ids = prompt_node_type_ids[:prompt_len]
        answer_len = input_ids.shape[0] - prompt_len
        node_type_ids = torch.tensor(
            prompt_node_type_ids + [OTHER_NODE_TYPE_ID] * answer_len,
            dtype=torch.long,
        )

        assert input_ids.shape == attention_mask.shape == labels.shape == node_type_ids.shape
        if prompt_len > 0:
            assert torch.all(labels[:prompt_len] == -100)
        if prompt_len < labels.shape[0]:
            # At least one answer token should remain trainable when the example is not fully truncated.
            assert torch.any(labels[prompt_len:] != -100)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "node_type_ids": node_type_ids,
            "prompt": example.prompt,
            "output": example.output,
            "prompt_symbolic_tokens": symbolic_tokens,
            "prompt_node_type_names": node_type_names,
        }

    def take(self, count: int | None) -> "StructuredJsonlDataset":
        """Return a lightweight prefix subset without re-reading the file."""

        if count is None or count >= len(self.examples):
            return self
        subset = StructuredJsonlDataset.__new__(StructuredJsonlDataset)
        subset.path = self.path
        subset.tokenizer = self.tokenizer
        subset.max_length = self.max_length
        subset.examples = self.examples[:count]
        return subset


class StructuredCollator:
    """Pad structured examples so every tensor in a batch has the same shape."""

    def __init__(self, tokenizer: Any):
        self.pad_token_id = tokenizer.pad_token_id
        if self.pad_token_id is None:
            raise ValueError("Tokenizer must define pad_token_id before batching.")

    def __call__(self, batch: list[dict[str, torch.Tensor | str | list[str]]]) -> dict[str, Any]:
        max_len = max(item["input_ids"].shape[0] for item in batch)

        input_ids = []
        attention_masks = []
        labels = []
        node_type_ids = []
        prompts = []
        outputs = []
        prompt_symbolic_tokens = []
        prompt_node_type_names = []

        for item in batch:
            # Validate per-example shapes before padding so mistakes fail early.
            assert item["input_ids"].shape == item["attention_mask"].shape
            assert item["input_ids"].shape == item["labels"].shape
            assert item["input_ids"].shape == item["node_type_ids"].shape

            pad_len = max_len - item["input_ids"].shape[0]
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
            node_type_ids.append(
                torch.cat(
                    [
                        item["node_type_ids"],
                        # Padded positions are never attended to, so OTHER is a safe fill value here.
                        torch.full((pad_len,), OTHER_NODE_TYPE_ID, dtype=torch.long),
                    ]
                )
            )
            prompts.append(item["prompt"])
            outputs.append(item["output"])
            prompt_symbolic_tokens.append(item["prompt_symbolic_tokens"])
            prompt_node_type_names.append(item["prompt_node_type_names"])

        batch_dict = {
            "input_ids": torch.stack(input_ids),
            "attention_mask": torch.stack(attention_masks),
            "labels": torch.stack(labels),
            "node_type_ids": torch.stack(node_type_ids),
            "prompts": prompts,
            "outputs": outputs,
            "prompt_symbolic_tokens": prompt_symbolic_tokens,
            "prompt_node_type_names": prompt_node_type_names,
        }

        # The structured model expects node_type_ids and input_ids to stay perfectly aligned.
        assert batch_dict["input_ids"].shape == batch_dict["node_type_ids"].shape
        assert batch_dict["input_ids"].shape == batch_dict["labels"].shape
        return batch_dict


__all__ = [
    "ID_TO_NODE_TYPE",
    "StructuredAlgebraExample",
    "StructuredCollator",
    "StructuredJsonlDataset",
    "align_prompt_node_type_ids",
]
