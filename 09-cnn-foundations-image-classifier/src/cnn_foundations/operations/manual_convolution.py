from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from cnn_foundations.operations.shapes import conv_output_size


@dataclass(frozen=True)
class CorrelationStep:
    output_position: tuple[int, int]
    window: list[list[float]]
    products: list[list[float]]
    total: float


def cross_correlate_2d(
    matrix: NDArray[np.float64],
    kernel: NDArray[np.float64],
    *,
    stride: int = 1,
    padding: int = 0,
    trace_limit: int = 16,
) -> tuple[NDArray[np.float64], list[dict[str, Any]]]:
    image = np.asarray(matrix, dtype=np.float64)
    weights = np.asarray(kernel, dtype=np.float64)
    if image.ndim != 2:
        raise ValueError("educational input must be a two-dimensional matrix.")
    if weights.shape != (3, 3):
        raise ValueError("educational kernel must have shape [3, 3].")
    if image.shape[0] > 12 or image.shape[1] > 12:
        raise ValueError("educational matrices are limited to 12 by 12.")
    if stride not in (1, 2):
        raise ValueError("stride must be 1 or 2.")
    if padding not in (0, 1):
        raise ValueError("padding must be 0 or 1.")

    padded = np.pad(image, padding, mode="constant")
    out_h = conv_output_size(image.shape[0], 3, stride=stride, padding=padding)
    out_w = conv_output_size(image.shape[1], 3, stride=stride, padding=padding)
    output = np.empty((out_h, out_w), dtype=np.float64)
    steps: list[dict[str, Any]] = []

    for row in range(out_h):
        for column in range(out_w):
            row_start = row * stride
            column_start = column * stride
            window = padded[row_start : row_start + 3, column_start : column_start + 3]
            products = window * weights
            total = float(products.sum())
            output[row, column] = total
            if len(steps) < trace_limit:
                step = CorrelationStep(
                    output_position=(row, column),
                    window=window.tolist(),
                    products=products.tolist(),
                    total=total,
                )
                steps.append(asdict(step))
    return output, steps
