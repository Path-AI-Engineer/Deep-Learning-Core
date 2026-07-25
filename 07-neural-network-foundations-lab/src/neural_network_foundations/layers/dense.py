"""A fully inspectable dense layer."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


@dataclass
class Dense:
    input_units: int
    output_units: int
    weights: FloatArray
    bias: FloatArray
    layer_id: str
    input_cache: FloatArray | None = None
    grad_weights: FloatArray | None = None
    grad_bias: FloatArray | None = None

    def __post_init__(self) -> None:
        self.weights = np.asarray(self.weights, dtype=np.float64)
        self.bias = np.asarray(self.bias, dtype=np.float64)
        if self.input_units < 1 or self.output_units < 1:
            raise ValueError("Dense dimensions must be positive.")
        if self.weights.shape != (self.input_units, self.output_units):
            raise ValueError(
                f"weights must have shape {(self.input_units, self.output_units)}, "
                f"received {self.weights.shape}."
            )
        if self.bias.shape != (1, self.output_units):
            raise ValueError(
                f"bias must have shape {(1, self.output_units)}, received {self.bias.shape}."
            )
        if not np.isfinite(self.weights).all() or not np.isfinite(self.bias).all():
            raise ValueError("Dense parameters must be finite.")

    @classmethod
    def small_normal(
        cls,
        input_units: int,
        output_units: int,
        *,
        rng: np.random.Generator,
        layer_id: str,
        scale: float = 0.1,
    ) -> Dense:
        return cls(
            input_units=input_units,
            output_units=output_units,
            weights=rng.normal(0.0, scale, (input_units, output_units)),
            bias=np.zeros((1, output_units), dtype=np.float64),
            layer_id=layer_id,
        )

    def forward(self, values: FloatArray, *, cache: bool = True) -> FloatArray:
        inputs = np.asarray(values, dtype=np.float64)
        if inputs.ndim != 2 or inputs.shape[1] != self.input_units:
            raise ValueError(f"inputs must have shape (samples, {self.input_units}).")
        if not np.isfinite(inputs).all():
            raise ValueError("Dense inputs must be finite.")
        if cache:
            self.input_cache = inputs.copy()
        return inputs @ self.weights + self.bias

    def backward(self, upstream_gradient: FloatArray) -> FloatArray:
        if self.input_cache is None:
            raise RuntimeError("forward must run before backward.")
        upstream = np.asarray(upstream_gradient, dtype=np.float64)
        expected = (self.input_cache.shape[0], self.output_units)
        if upstream.shape != expected:
            raise ValueError(f"upstream_gradient must have shape {expected}.")
        self.grad_weights = self.input_cache.T @ upstream
        self.grad_bias = np.sum(upstream, axis=0, keepdims=True)
        return upstream @ self.weights.T
