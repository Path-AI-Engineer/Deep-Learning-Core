from __future__ import annotations

import torch
from torch import nn


class FashionMLP(nn.Module):
    def __init__(self, *, hidden_features: int = 128, dropout: float = 0.25) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, hidden_features),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_features, 10),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 4 or tuple(inputs.shape[1:]) != (1, 28, 28):
            raise ValueError("FashionMLP expects an NCHW tensor shaped [N, 1, 28, 28].")
        output: torch.Tensor = self.network(inputs)
        return output
