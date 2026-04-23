"""Structured Pythia model with learned node-type embeddings."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from transformers import GPTNeoXForCausalLM

try:
    from .node_types import NUM_NODE_TYPES
except ImportError:
    from node_types import NUM_NODE_TYPES


class StructuredPythia(nn.Module):
    """Wrap GPTNeoXForCausalLM and add one learned node-type embedding per token."""

    def __init__(self, model_name: str, freeze_base: bool = True):
        super().__init__()
        self.model_name = model_name
        self.freeze_base = freeze_base
        # Load the base causal LM exactly once and keep its architecture unchanged.
        self.base = GPTNeoXForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)
        hidden_size = self.base.config.hidden_size
        # One learned vector per node type, matched to the LM hidden size.
        self.node_type_embedding = nn.Embedding(NUM_NODE_TYPES, hidden_size)

        # Zero init keeps the first forward pass identical to the frozen baseline.
        nn.init.zeros_(self.node_type_embedding.weight)

        if freeze_base:
            # Baseline structured training only updates the added node-type embedding table.
            for parameter in self.base.parameters():
                parameter.requires_grad = False

    def forward(
        self,
        input_ids: torch.Tensor,
        node_type_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ):
        # Reuse the base model's token embedding layer so vocabulary handling stays identical.
        token_embeds = self.base.get_input_embeddings()(input_ids)
        # Look up one structural embedding per token position.
        node_type_embeds = self.node_type_embedding(node_type_ids)
        # Inject structure by summing token and node-type embeddings position-wise.
        inputs_embeds = token_embeds + node_type_embeds
        return self.base(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
        )

    def save_pretrained(self, save_dir: str | Path) -> Path:
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        # Save the base LM in standard Hugging Face format.
        self.base.save_pretrained(save_path)
        # Save the extra structured weights separately because they are not part of GPTNeoX itself.
        torch.save(
            {
                "node_type_embedding": self.node_type_embedding.state_dict(),
                "freeze_base": self.freeze_base,
            },
            save_path / "structured_state.pt",
        )
        return save_path

    @classmethod
    def from_checkpoint(cls, checkpoint_dir: str | Path, freeze_base: bool = True) -> "StructuredPythia":
        checkpoint_path = Path(checkpoint_dir)
        # Rebuild the wrapper from the saved base-model directory.
        model = cls(str(checkpoint_path), freeze_base=freeze_base)
        state_path = checkpoint_path / "structured_state.pt"
        if state_path.exists():
            # Restore the learned node-type embedding table when present.
            state = torch.load(state_path, map_location="cpu")
            model.node_type_embedding.load_state_dict(state["node_type_embedding"])
        return model
