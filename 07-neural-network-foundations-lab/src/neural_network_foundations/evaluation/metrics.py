"""Small, explicit evaluation metrics."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def binary_accuracy(predictions: FloatArray, targets: FloatArray) -> float:
    prediction = np.asarray(predictions, dtype=np.float64)
    target = np.asarray(targets, dtype=np.float64)
    if prediction.shape != target.shape:
        raise ValueError("predictions and targets must have the same shape.")
    labels = (prediction >= 0.5).astype(np.float64)
    return float(np.mean(labels == target))
