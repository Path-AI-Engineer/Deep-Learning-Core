from __future__ import annotations

import torch
from torch import nn


class EducationalLayerNorm(nn.Module):
    def __init__(self, features: int, epsilon: float = 1e-5) -> None:
        super().__init__()
        self.epsilon = epsilon
        self.gamma = nn.Parameter(torch.ones(features))
        self.beta = nn.Parameter(torch.zeros(features))

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        mean = tensor.mean(dim=-1, keepdim=True)
        variance = tensor.var(dim=-1, unbiased=False, keepdim=True)
        normalized = (tensor - mean) / torch.sqrt(variance + self.epsilon)
        return self.gamma * normalized + self.beta

