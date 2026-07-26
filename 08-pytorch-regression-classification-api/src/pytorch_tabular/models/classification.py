from __future__ import annotations

import torch
from torch import nn


class ClassificationMLP(nn.Module):
    def __init__(
        self,
        input_features: int,
        class_count: int,
        hidden_units: tuple[int, ...] = (48, 24),
        dropout: float = 0.10,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        previous = input_features
        for width in hidden_units:
            layers.extend([nn.Linear(previous, width), nn.ReLU(), nn.Dropout(dropout)])
            previous = width
        layers.append(nn.Linear(previous, class_count))
        self.network = nn.Sequential(*layers)
        self.input_features = input_features
        self.class_count = class_count
        self.hidden_units = hidden_units
        self.dropout = dropout

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features)
