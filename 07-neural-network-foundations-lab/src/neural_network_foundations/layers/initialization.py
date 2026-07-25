"""Pedagogical parameter initialization strategies."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def initialize_parameters(
    input_units: int,
    output_units: int,
    *,
    strategy: str,
    rng: np.random.Generator,
) -> tuple[FloatArray, FloatArray]:
    if input_units < 1 or output_units < 1:
        raise ValueError("Layer dimensions must be positive.")
    if strategy == "zeros":
        weights = np.zeros((input_units, output_units), dtype=np.float64)
    elif strategy == "small_normal":
        weights = rng.normal(0.0, 0.1, (input_units, output_units))
    elif strategy == "xavier":
        scale = np.sqrt(2.0 / (input_units + output_units))
        weights = rng.normal(0.0, scale, (input_units, output_units))
    elif strategy == "he":
        scale = np.sqrt(2.0 / input_units)
        weights = rng.normal(0.0, scale, (input_units, output_units))
    else:
        raise ValueError(f"Unsupported initialization strategy: {strategy}.")
    return weights.astype(np.float64), np.zeros((1, output_units), dtype=np.float64)
