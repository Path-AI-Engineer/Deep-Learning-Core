from __future__ import annotations

import torch
from torch import nn


class RegressionMLP(nn.Module):
    def __init__(
        self,
        input_features: int,
        hidden_units: tuple[int, ...] = (64, 32),
        dropout: float = 0.10,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        previous = input_features
        for width in hidden_units:
            layers.extend([nn.Linear(previous, width), nn.ReLU(), nn.Dropout(dropout)])
            previous = width
        layers.append(nn.Linear(previous, 1))
        self.network = nn.Sequential(*layers)
        self.input_features = input_features
        self.hidden_units = hidden_units
        self.dropout = dropout

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features).squeeze(-1)
