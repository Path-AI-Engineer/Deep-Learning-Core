"""Bounded 2D decision-grid generation."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from neural_network_foundations.models import MLP

FloatArray = NDArray[np.float64]


def decision_boundary(
    model: MLP,
    features: FloatArray,
    *,
    resolution: int,
    padding: float = 0.25,
) -> dict[str, Any]:
    if not 10 <= resolution <= 100:
        raise ValueError("resolution must be between 10 and 100.")
    values = np.asarray(features, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 2:
        raise ValueError("features must have shape (samples, 2).")
    x_values = np.linspace(values[:, 0].min() - padding, values[:, 0].max() + padding, resolution)
    y_values = np.linspace(values[:, 1].min() - padding, values[:, 1].max() + padding, resolution)
    mesh_x, mesh_y = np.meshgrid(x_values, y_values)
    points = np.column_stack((mesh_x.ravel(), mesh_y.ravel()))
    probabilities = model.forward(points, cache=False).reshape(resolution, resolution)
    return {
        "resolution": resolution,
        "x": x_values.tolist(),
        "y": y_values.tolist(),
        "probabilities": probabilities.tolist(),
    }
