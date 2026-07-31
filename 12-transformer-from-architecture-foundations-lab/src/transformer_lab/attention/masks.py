from __future__ import annotations

import torch


def causal_mask(length: int, *, device: torch.device | None = None) -> torch.Tensor:
    """Return a boolean mask where True means blocked."""
    if length < 1:
        raise ValueError("Causal mask length must be positive.")
    return torch.triu(
        torch.ones((length, length), dtype=torch.bool, device=device),
        diagonal=1,
    )


def combine_masks(
    *,
    batch_size: int,
    query_length: int,
    key_length: int,
    key_padding_mask: torch.Tensor | None = None,
    attention_mask: torch.Tensor | None = None,
) -> torch.Tensor | None:
    combined: torch.Tensor | None = None
    if key_padding_mask is not None:
        if key_padding_mask.shape != (batch_size, key_length):
            raise ValueError("key_padding_mask must have shape [B, T_key].")
        combined = key_padding_mask[:, None, None, :].expand(
            batch_size, 1, query_length, key_length
        )
    if attention_mask is not None:
        if attention_mask.shape != (query_length, key_length):
            raise ValueError("attention_mask must have shape [T_query, T_key].")
        expanded = attention_mask[None, None, :, :].expand(
            batch_size, 1, query_length, key_length
        )
        combined = expanded if combined is None else combined | expanded
    if combined is not None and combined.all(dim=-1).any():
        raise ValueError("Attention contains a fully masked query row.")
    return combined

