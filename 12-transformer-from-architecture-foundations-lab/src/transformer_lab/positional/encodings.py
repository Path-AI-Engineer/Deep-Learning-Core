from __future__ import annotations

import math
from typing import cast

import torch
from torch import nn

from transformer_lab.contracts import PositionKind


class SinusoidalPosition(nn.Module):
    def __init__(self, d_model: int, max_length: int) -> None:
        super().__init__()
        position = torch.arange(max_length, dtype=torch.float32).unsqueeze(1)
        denominator = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10_000.0) / d_model)
        )
        encoding = torch.zeros(max_length, d_model)
        encoding[:, 0::2] = torch.sin(position * denominator)
        encoding[:, 1::2] = torch.cos(position * denominator[: encoding[:, 1::2].shape[1]])
        self.encoding: torch.Tensor
        self.register_buffer("encoding", encoding, persistent=True)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        if tensor.shape[1] > self.encoding.shape[0]:
            raise ValueError("Sequence exceeds configured positional capacity.")
        return tensor + self.encoding[: tensor.shape[1]].to(tensor.dtype)


class LearnedPosition(nn.Module):
    def __init__(self, d_model: int, max_length: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(max_length, d_model)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        if tensor.shape[1] > self.embedding.num_embeddings:
            raise ValueError("Sequence exceeds configured positional capacity.")
        positions = torch.arange(tensor.shape[1], device=tensor.device)
        return cast(torch.Tensor, tensor + self.embedding(positions)[None, :, :])


class NoPosition(nn.Module):
    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor


def build_position(kind: PositionKind, d_model: int, max_length: int) -> nn.Module:
    if kind == "sinusoidal":
        return SinusoidalPosition(d_model, max_length)
    if kind == "learned":
        return LearnedPosition(d_model, max_length)
    return NoPosition()
