from __future__ import annotations

import math


def conv_output_size(
    input_size: int,
    kernel_size: int,
    *,
    stride: int = 1,
    padding: int = 0,
    dilation: int = 1,
) -> int:
    values = (input_size, kernel_size, stride, dilation)
    if any(value <= 0 for value in values):
        raise ValueError("input, kernel, stride and dilation must be positive.")
    if padding < 0:
        raise ValueError("padding cannot be negative.")
    numerator = input_size + (2 * padding) - dilation * (kernel_size - 1) - 1
    output = math.floor(numerator / stride + 1)
    if output <= 0:
        raise ValueError("configuration produces an empty output.")
    return output


def validate_nchw(shape: tuple[int, ...]) -> None:
    if len(shape) != 4:
        raise ValueError("image tensors must use NCHW rank four.")
    batch, channels, height, width = shape
    if min(batch, channels, height, width) <= 0:
        raise ValueError("NCHW dimensions must be positive.")

