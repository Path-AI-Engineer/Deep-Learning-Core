from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as functional

from transformer_lab.attention.masks import combine_masks
from transformer_lab.attention.scaled import scaled_dot_product_attention


class MultiHeadAttention(nn.Module):
    """Multi-head attention implemented from local projection and scoring primitives."""

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        if d_model % num_heads:
            raise ValueError("d_model must be divisible by num_heads.")
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.dropout = dropout
        self.query_projection = nn.Linear(d_model, d_model)
        self.key_projection = nn.Linear(d_model, d_model)
        self.value_projection = nn.Linear(d_model, d_model)
        self.output_projection = nn.Linear(d_model, d_model)

    def split_heads(self, tensor: torch.Tensor) -> torch.Tensor:
        batch, length, _ = tensor.shape
        return (
            tensor.reshape(batch, length, self.num_heads, self.head_dim)
            .transpose(1, 2)
            .contiguous()
        )

    def merge_heads(self, tensor: torch.Tensor) -> torch.Tensor:
        batch, _, length, _ = tensor.shape
        return (
            tensor.transpose(1, 2)
            .contiguous()
            .reshape(batch, length, self.d_model)
        )

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        key_padding_mask: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        need_weights: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if query.ndim != 3 or key.ndim != 3 or value.ndim != 3:
            raise ValueError("Multi-head inputs must have shape [B, T, D].")
        batch, query_length, _ = query.shape
        key_length = key.shape[1]
        mask = combine_masks(
            batch_size=batch,
            query_length=query_length,
            key_length=key_length,
            key_padding_mask=key_padding_mask,
            attention_mask=attention_mask,
        )
        q = self.split_heads(self.query_projection(query))
        k = self.split_heads(self.key_projection(key))
        v = self.split_heads(self.value_projection(value))
        _, weights, _ = scaled_dot_product_attention(q, k, v, mask=mask)
        weights = functional.dropout(weights, self.dropout, self.training)
        output = weights @ v
        projected = self.output_projection(self.merge_heads(output))
        return projected, weights if need_weights else None
