from __future__ import annotations

import math
from typing import Any, cast

import torch
from torch import nn

from transformer_lab.attention import causal_mask
from transformer_lab.blocks import DecoderBlock, EncoderBlock
from transformer_lab.contracts import ModelConfig
from transformer_lab.positional import build_position


class SequenceTransformer(nn.Module):
    """Encoder-decoder Transformer assembled from project-local primitives."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.source_embedding = nn.Embedding(config.vocabulary_size, config.d_model)
        self.target_embedding = nn.Embedding(config.vocabulary_size, config.d_model)
        self.source_position = build_position(
            config.positional_encoding, config.d_model, config.max_length
        )
        self.target_position = build_position(
            config.positional_encoding, config.d_model, config.max_length
        )
        self.encoder = nn.ModuleList(
            EncoderBlock(config) for _ in range(config.encoder_layers)
        )
        self.decoder = nn.ModuleList(
            DecoderBlock(config) for _ in range(config.decoder_layers)
        )
        self.output_projection = nn.Linear(config.d_model, config.vocabulary_size)
        self.dropout = nn.Dropout(config.dropout)

    def _embed(self, ids: torch.Tensor, *, source: bool) -> torch.Tensor:
        embedding = self.source_embedding if source else self.target_embedding
        position = self.source_position if source else self.target_position
        tensor = embedding(ids) * math.sqrt(self.config.d_model)
        return cast(torch.Tensor, self.dropout(position(tensor)))

    def encode(
        self,
        source_ids: torch.Tensor,
        source_padding_mask: torch.Tensor | None,
        *,
        trace: bool = False,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        tensor = self._embed(source_ids, source=True)
        weights: list[torch.Tensor] = []
        for layer in self.encoder:
            tensor, attention = layer(
                tensor,
                padding_mask=source_padding_mask,
                need_weights=trace,
            )
            if attention is not None:
                weights.append(attention)
        return tensor, weights

    def decode(
        self,
        target_ids: torch.Tensor,
        memory: torch.Tensor,
        *,
        target_padding_mask: torch.Tensor | None,
        source_padding_mask: torch.Tensor | None,
        trace: bool = False,
    ) -> tuple[torch.Tensor, dict[str, list[torch.Tensor]]]:
        tensor = self._embed(target_ids, source=False)
        mask = causal_mask(target_ids.shape[1], device=target_ids.device)
        self_weights: list[torch.Tensor] = []
        cross_weights: list[torch.Tensor] = []
        for layer in self.decoder:
            tensor, self_attention, cross_attention = layer(
                tensor,
                memory,
                target_padding_mask=target_padding_mask,
                source_padding_mask=source_padding_mask,
                causal=mask,
                need_weights=trace,
            )
            if self_attention is not None:
                self_weights.append(self_attention)
            if cross_attention is not None:
                cross_weights.append(cross_attention)
        return self.output_projection(tensor), {
            "decoder_self": self_weights,
            "cross": cross_weights,
        }

    def forward(
        self,
        source_ids: torch.Tensor,
        target_ids: torch.Tensor,
        *,
        source_padding_mask: torch.Tensor | None = None,
        target_padding_mask: torch.Tensor | None = None,
        trace: bool = False,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        memory, encoder_weights = self.encode(
            source_ids, source_padding_mask, trace=trace
        )
        logits, decoder_weights = self.decode(
            target_ids,
            memory,
            target_padding_mask=target_padding_mask,
            source_padding_mask=source_padding_mask,
            trace=trace,
        )
        return logits, {
            "encoder_self": encoder_weights,
            **decoder_weights,
            "memory_shape": list(memory.shape),
            "logits_shape": list(logits.shape),
        }
