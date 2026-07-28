from __future__ import annotations

from typing import Any

import torch
from torch import nn


class FashionCNN(nn.Module):
    def __init__(
        self,
        *,
        channels: tuple[int, int] = (16, 32),
        hidden_features: int = 128,
        dropout: float = 0.25,
        batch_norm: bool = True,
    ) -> None:
        super().__init__()
        first, second = channels
        self.conv1 = nn.Conv2d(1, first, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(first) if batch_norm else nn.Identity()
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(first, second, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(second) if batch_norm else nn.Identity()
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(second * 7 * 7, hidden_features),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_features, 10),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 4 or tuple(inputs.shape[1:]) != (1, 28, 28):
            raise ValueError("FashionCNN expects an NCHW tensor shaped [N, 1, 28, 28].")
        value = self.pool1(self.relu1(self.bn1(self.conv1(inputs))))
        value = self.pool2(self.relu2(self.bn2(self.conv2(value))))
        output: torch.Tensor = self.classifier(value)
        return output

    def shape_trace(self, batch_size: int = 1) -> list[dict[str, Any]]:
        trace: list[dict[str, Any]] = []
        value = torch.zeros(batch_size, 1, 28, 28)
        for name in ("conv1", "bn1", "relu1", "pool1", "conv2", "bn2", "relu2", "pool2"):
            value = getattr(self, name)(value)
            trace.append({"layer": name, "shape": list(value.shape)})
        logits = self.classifier(value)
        trace.append({"layer": "classifier", "shape": list(logits.shape)})
        return trace


def model_from_config(config: dict[str, Any]) -> FashionCNN:
    channels = tuple(int(value) for value in config.get("conv_channels", (16, 32)))
    if len(channels) != 2:
        raise ValueError("CNN bundle requires two convolution channel widths.")
    return FashionCNN(
        channels=(channels[0], channels[1]),
        hidden_features=int(config.get("hidden_features", 128)),
        dropout=float(config.get("dropout", 0.25)),
        batch_norm=bool(config.get("batch_norm", True)),
    )
