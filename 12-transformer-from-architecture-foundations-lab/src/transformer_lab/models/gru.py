from __future__ import annotations

from typing import cast

import torch
from torch import nn

from transformer_lab.contracts import ModelConfig


class GRUEncoderDecoder(nn.Module):
    """Local recurrent baseline using the same vocabulary and teacher forcing."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.source_embedding = nn.Embedding(config.vocabulary_size, config.d_model)
        self.target_embedding = nn.Embedding(config.vocabulary_size, config.d_model)
        self.encoder = nn.GRU(config.d_model, config.d_model, batch_first=True)
        self.decoder = nn.GRU(config.d_model, config.d_model, batch_first=True)
        self.output_projection = nn.Linear(config.d_model, config.vocabulary_size)

    def forward(
        self,
        source_ids: torch.Tensor,
        target_ids: torch.Tensor,
    ) -> torch.Tensor:
        _, hidden = self.encoder(self.source_embedding(source_ids))
        output, _ = self.decoder(self.target_embedding(target_ids), hidden)
        return cast(torch.Tensor, self.output_projection(output))
