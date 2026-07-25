"""Stable scalar losses for small supervised experiments."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
EPSILON = 1e-12


def _pair(predictions: FloatArray, targets: FloatArray) -> tuple[FloatArray, FloatArray]:
    prediction = np.asarray(predictions, dtype=np.float64)
    target = np.asarray(targets, dtype=np.float64)
    if prediction.shape != target.shape or prediction.ndim != 2:
        raise ValueError("predictions and targets must have the same two-dimensional shape.")
    if not np.isfinite(prediction).all() or not np.isfinite(target).all():
        raise ValueError("loss inputs must be finite.")
    return prediction, target


def mean_squared_error(predictions: FloatArray, targets: FloatArray) -> float:
    prediction, target = _pair(predictions, targets)
    return float(np.mean((prediction - target) ** 2))


def binary_cross_entropy(predictions: FloatArray, targets: FloatArray) -> float:
    prediction, target = _pair(predictions, targets)
    if not np.isin(target, [0.0, 1.0]).all():
        raise ValueError("Binary Cross-Entropy targets must be zero or one.")
    clipped = np.clip(prediction, EPSILON, 1.0 - EPSILON)
    return float(-np.mean(target * np.log(clipped) + (1.0 - target) * np.log(1.0 - clipped)))


def loss(name: str, predictions: FloatArray, targets: FloatArray) -> float:
    if name == "mean_squared_error":
        return mean_squared_error(predictions, targets)
    if name == "binary_cross_entropy":
        return binary_cross_entropy(predictions, targets)
    raise ValueError(f"Unsupported loss: {name}.")


def loss_derivative(name: str, predictions: FloatArray, targets: FloatArray) -> FloatArray:
    prediction, target = _pair(predictions, targets)
    count = prediction.size
    if name == "mean_squared_error":
        return 2.0 * (prediction - target) / count
    if name == "binary_cross_entropy":
        clipped = np.clip(prediction, EPSILON, 1.0 - EPSILON)
        return (-(target / clipped) + (1.0 - target) / (1.0 - clipped)) / count
    raise ValueError(f"Unsupported loss: {name}.")
