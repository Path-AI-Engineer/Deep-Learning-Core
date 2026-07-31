from __future__ import annotations

from typing import cast

import torch
from torch import nn

from transformer_lab.attention import MultiHeadAttention
from transformer_lab.contracts import ModelConfig


class PositionwiseFeedForward(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        activation: nn.Module = nn.GELU() if config.activation == "gelu" else nn.ReLU()
        self.network = nn.Sequential(
            nn.Linear(config.d_model, config.d_ff),
            activation,
            nn.Dropout(config.dropout),
            nn.Linear(config.d_ff, config.d_model),
        )

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, self.network(tensor))


class EncoderBlock(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.normalization = config.normalization
        self.attention = MultiHeadAttention(
            config.d_model, config.num_heads, config.dropout
        )
        self.feed_forward = PositionwiseFeedForward(config)
        self.norm_attention = nn.LayerNorm(config.d_model)
        self.norm_feed_forward = nn.LayerNorm(config.d_model)
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        tensor: torch.Tensor,
        *,
        padding_mask: torch.Tensor | None,
        need_weights: bool,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if self.normalization == "pre":
            normalized = self.norm_attention(tensor)
            attention, weights = self.attention(
                normalized,
                normalized,
                normalized,
                key_padding_mask=padding_mask,
                need_weights=need_weights,
            )
            tensor = tensor + self.dropout(attention)
            tensor = tensor + self.dropout(
                self.feed_forward(self.norm_feed_forward(tensor))
            )
            return tensor, weights
        attention, weights = self.attention(
            tensor,
            tensor,
            tensor,
            key_padding_mask=padding_mask,
            need_weights=need_weights,
        )
        tensor = self.norm_attention(tensor + self.dropout(attention))
        tensor = self.norm_feed_forward(tensor + self.dropout(self.feed_forward(tensor)))
        return tensor, weights


class DecoderBlock(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.normalization = config.normalization
        self.self_attention = MultiHeadAttention(
            config.d_model, config.num_heads, config.dropout
        )
        self.cross_attention = MultiHeadAttention(
            config.d_model, config.num_heads, config.dropout
        )
        self.feed_forward = PositionwiseFeedForward(config)
        self.norm_self = nn.LayerNorm(config.d_model)
        self.norm_cross = nn.LayerNorm(config.d_model)
        self.norm_feed_forward = nn.LayerNorm(config.d_model)
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        tensor: torch.Tensor,
        memory: torch.Tensor,
        *,
        target_padding_mask: torch.Tensor | None,
        source_padding_mask: torch.Tensor | None,
        causal: torch.Tensor,
        need_weights: bool,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        if self.normalization == "pre":
            normalized = self.norm_self(tensor)
            self_output, self_weights = self.self_attention(
                normalized,
                normalized,
                normalized,
                key_padding_mask=target_padding_mask,
                attention_mask=causal,
                need_weights=need_weights,
            )
            tensor = tensor + self.dropout(self_output)
            cross_query = self.norm_cross(tensor)
            cross_output, cross_weights = self.cross_attention(
                cross_query,
                memory,
                memory,
                key_padding_mask=source_padding_mask,
                need_weights=need_weights,
            )
            tensor = tensor + self.dropout(cross_output)
            tensor = tensor + self.dropout(
                self.feed_forward(self.norm_feed_forward(tensor))
            )
            return tensor, self_weights, cross_weights
        self_output, self_weights = self.self_attention(
            tensor,
            tensor,
            tensor,
            key_padding_mask=target_padding_mask,
            attention_mask=causal,
            need_weights=need_weights,
        )
        tensor = self.norm_self(tensor + self.dropout(self_output))
        cross_output, cross_weights = self.cross_attention(
            tensor,
            memory,
            memory,
            key_padding_mask=source_padding_mask,
            need_weights=need_weights,
        )
        tensor = self.norm_cross(tensor + self.dropout(cross_output))
        tensor = self.norm_feed_forward(tensor + self.dropout(self.feed_forward(tensor)))
        return tensor, self_weights, cross_weights
