from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn.functional as functional

from cnn_foundations.operations.manual_convolution import cross_correlate_2d
from cnn_foundations.operations.receptive_field import receptive_field_trace
from cnn_foundations.operations.shapes import conv_output_size


@pytest.mark.parametrize(("stride", "padding"), [(1, 0), (1, 1), (2, 0), (2, 1)])
def test_manual_cross_correlation_matches_torch(stride: int, padding: int) -> None:
    matrix = np.arange(36, dtype=np.float64).reshape(6, 6)
    kernel = np.asarray([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=np.float64)
    actual, _ = cross_correlate_2d(matrix, kernel, stride=stride, padding=padding)
    expected = functional.conv2d(
        torch.tensor(matrix)[None, None],
        torch.tensor(kernel)[None, None],
        stride=stride,
        padding=padding,
    )[0, 0].numpy()
    np.testing.assert_allclose(actual, expected)


def test_convolution_shape_formula() -> None:
    assert conv_output_size(28, kernel_size=3, stride=1, padding=1) == 28
    assert conv_output_size(28, kernel_size=2, stride=2, padding=0) == 14


def test_invalid_convolution_shape_is_rejected() -> None:
    with pytest.raises(ValueError):
        conv_output_size(2, kernel_size=5, stride=1, padding=0)


def test_receptive_field_trace_grows() -> None:
    trace = receptive_field_trace(
        (("conv1", 3, 1), ("pool1", 2, 2), ("conv2", 3, 1), ("pool2", 2, 2))
    )
    assert trace[-1]["receptive_field"] == 10
    assert trace[-1]["jump"] == 4
