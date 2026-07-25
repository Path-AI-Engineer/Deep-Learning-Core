"""Numerically stable activation functions."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def sigmoid(values: FloatArray) -> FloatArray:
    x = np.asarray(values, dtype=np.float64)
    result = np.empty_like(x)
    positive = x >= 0
    result[positive] = 1.0 / (1.0 + np.exp(-x[positive]))
    exp_x = np.exp(x[~positive])
    result[~positive] = exp_x / (1.0 + exp_x)
    return result


def tanh(values: FloatArray) -> FloatArray:
    return np.tanh(np.asarray(values, dtype=np.float64))


def relu(values: FloatArray) -> FloatArray:
    return np.maximum(np.asarray(values, dtype=np.float64), 0.0)


def activation(name: str, values: FloatArray) -> FloatArray:
    functions = {"sigmoid": sigmoid, "tanh": tanh, "relu": relu}
    try:
        return functions[name](values)
    except KeyError as exc:
        raise ValueError(f"Unsupported activation: {name}.") from exc


def derivative(name: str, preactivation: FloatArray) -> FloatArray:
    z = np.asarray(preactivation, dtype=np.float64)
    if name == "sigmoid":
        output = sigmoid(z)
        return output * (1.0 - output)
    if name == "tanh":
        output = tanh(z)
        return 1.0 - output**2
    if name == "relu":
        return (z > 0.0).astype(np.float64)
    raise ValueError(f"Unsupported activation: {name}.")
