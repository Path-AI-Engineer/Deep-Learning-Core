from __future__ import annotations

import math

import torch


def stable_softmax(values: torch.Tensor, *, dim: int = -1) -> torch.Tensor:
    if not torch.isfinite(values).all():
        raise ValueError("stable_softmax accepts finite values only.")
    shifted = values - values.amax(dim=dim, keepdim=True)
    numerator = shifted.exp()
    return numerator / numerator.sum(dim=dim, keepdim=True)


def scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Manual scaled attention for [B, H, T, D] tensors."""
    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
        raise ValueError("Q, K and V must have shape [B, H, T, D].")
    if query.shape[:2] != key.shape[:2] or key.shape[:3] != value.shape[:3]:
        raise ValueError("Q, K and V batch, head and key lengths are incompatible.")
    if query.shape[-1] != key.shape[-1]:
        raise ValueError("Q and K head dimensions must match.")
    scores = query @ key.transpose(-2, -1)
    scores = scores / math.sqrt(query.shape[-1])
    if mask is not None:
        try:
            expanded = torch.broadcast_to(mask, scores.shape)
        except RuntimeError as error:
            raise ValueError("Attention mask cannot broadcast to score shape.") from error
        if expanded.all(dim=-1).any():
            raise ValueError("Attention contains a fully masked query row.")
        scores = scores.masked_fill(expanded, -torch.inf)
    weights = torch.softmax(scores, dim=-1)
    if not torch.isfinite(weights).all():
        raise ValueError("Attention weights contain NaN or Inf.")
    return weights @ value, weights, scores

