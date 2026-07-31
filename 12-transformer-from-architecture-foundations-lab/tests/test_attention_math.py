from __future__ import annotations

import math

import pytest
import torch

from transformer_lab.attention import (
    MultiHeadAttention,
    causal_mask,
    scaled_dot_product_attention,
)
from transformer_lab.normalization import EducationalLayerNorm


def test_scaled_attention_matches_pytorch_reference() -> None:
    torch.manual_seed(12)
    query = torch.randn(2, 3, 5, 8, dtype=torch.float64)
    key = torch.randn(2, 3, 6, 8, dtype=torch.float64)
    value = torch.randn(2, 3, 6, 4, dtype=torch.float64)
    actual, weights, scores = scaled_dot_product_attention(query, key, value)
    expected_weights = torch.softmax(query @ key.transpose(-2, -1) / math.sqrt(8), dim=-1)
    assert torch.allclose(weights, expected_weights, atol=1e-10)
    assert torch.allclose(actual, expected_weights @ value, atol=1e-10)
    assert scores.shape == (2, 3, 5, 6)


def test_causal_mask_blocks_future_probability() -> None:
    query = torch.ones(1, 1, 4, 2)
    mask = causal_mask(4)
    _, weights, _ = scaled_dot_product_attention(query, query, query, mask=mask)
    assert torch.count_nonzero(torch.triu(weights[0, 0], diagonal=1)) == 0
    assert torch.allclose(weights.sum(dim=-1), torch.ones(1, 1, 4))


def test_multi_head_shapes_and_divisibility() -> None:
    layer = MultiHeadAttention(d_model=16, num_heads=4)
    output, weights = layer(
        torch.randn(2, 5, 16),
        torch.randn(2, 7, 16),
        torch.randn(2, 7, 16),
        need_weights=True,
    )
    assert output.shape == (2, 5, 16)
    assert weights is not None and weights.shape == (2, 4, 5, 7)
    with pytest.raises(ValueError, match="divisible"):
        MultiHeadAttention(d_model=15, num_heads=4)


def test_educational_layer_norm_has_zero_mean_unit_variance() -> None:
    normalization = EducationalLayerNorm(6, epsilon=1e-8)
    values = torch.randn(4, 5, 6)
    normalized = normalization(values)
    assert torch.allclose(normalized.mean(dim=-1), torch.zeros(4, 5), atol=1e-5)
    assert torch.allclose(
        normalized.var(dim=-1, unbiased=False),
        torch.ones(4, 5),
        atol=1e-4,
    )
