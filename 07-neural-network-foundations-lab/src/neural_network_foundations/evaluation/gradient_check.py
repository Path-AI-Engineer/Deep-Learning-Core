"""Central finite-difference validation for MLP parameter gradients."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from neural_network_foundations.losses import loss
from neural_network_foundations.models import MLP

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class ParameterGradientResult:
    parameter: str
    checked_values: int
    max_absolute_error: float
    max_relative_error: float
    passed: bool


@dataclass(frozen=True)
class GradientCheckReport:
    epsilon: float
    absolute_tolerance: float
    relative_tolerance: float
    passed: bool
    parameters: tuple[ParameterGradientResult, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _relative_error(analytical: FloatArray, numerical: FloatArray) -> FloatArray:
    denominator = np.maximum(1e-12, np.abs(analytical) + np.abs(numerical))
    return np.abs(analytical - numerical) / denominator


def check_model_gradients(
    model: MLP,
    features: FloatArray,
    targets: FloatArray,
    *,
    loss_name: str,
    epsilon: float = 1e-6,
    absolute_tolerance: float = 1e-6,
    relative_tolerance: float = 1e-4,
) -> GradientCheckReport:
    if epsilon <= 0:
        raise ValueError("epsilon must be positive.")
    model.forward(features)
    model.backward(targets, loss_name=loss_name)
    analytical = {name: value.copy() for name, value in model.gradients().items()}
    results: list[ParameterGradientResult] = []

    for name, parameter in model.parameters().items():
        numerical = np.zeros_like(parameter)
        for index in np.ndindex(parameter.shape):
            original = float(parameter[index])
            parameter[index] = original + epsilon
            plus_loss = loss(loss_name, model.forward(features, cache=False), targets)
            parameter[index] = original - epsilon
            minus_loss = loss(loss_name, model.forward(features, cache=False), targets)
            parameter[index] = original
            numerical[index] = (plus_loss - minus_loss) / (2.0 * epsilon)

        absolute = np.abs(analytical[name] - numerical)
        relative = _relative_error(analytical[name], numerical)
        parameter_passed = bool(
            np.all((absolute <= absolute_tolerance) | (relative <= relative_tolerance))
        )
        results.append(
            ParameterGradientResult(
                parameter=name,
                checked_values=parameter.size,
                max_absolute_error=float(np.max(absolute)),
                max_relative_error=float(np.max(relative)),
                passed=parameter_passed,
            )
        )

    model.forward(features)
    model.backward(targets, loss_name=loss_name)
    return GradientCheckReport(
        epsilon=epsilon,
        absolute_tolerance=absolute_tolerance,
        relative_tolerance=relative_tolerance,
        passed=all(item.passed for item in results),
        parameters=tuple(results),
    )
