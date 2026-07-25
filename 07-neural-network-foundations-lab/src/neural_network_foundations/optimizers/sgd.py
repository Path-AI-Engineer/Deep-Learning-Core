"""Explicit stochastic-gradient-descent parameter updates."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


class SGD:
    def __init__(self, learning_rate: float) -> None:
        if not 0.0001 <= learning_rate <= 1.0:
            raise ValueError("learning_rate must be between 0.0001 and 1.0.")
        self.learning_rate = learning_rate

    def step(
        self,
        parameters: dict[str, FloatArray],
        gradients: dict[str, FloatArray],
    ) -> None:
        if set(parameters) != set(gradients):
            raise ValueError("Parameters and gradients must have identical keys.")
        for name, parameter in parameters.items():
            gradient = gradients[name]
            if parameter.shape != gradient.shape:
                raise ValueError(f"Gradient shape mismatch for {name}.")
            if not np.isfinite(gradient).all():
                raise FloatingPointError(f"Non-finite gradient detected for {name}.")
            parameter -= self.learning_rate * gradient
            if not np.isfinite(parameter).all():
                raise FloatingPointError(f"Non-finite parameter detected for {name}.")
