"""Training diagnostics expressed as explainable signals."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from neural_network_foundations.models import MLP


@dataclass(frozen=True)
class Diagnostic:
    code: str
    severity: str
    signal: str
    explanation: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def diagnose(model: MLP, loss_history: list[float]) -> list[Diagnostic]:
    diagnostics: list[Diagnostic] = []
    if model.hidden_z is not None and model.config.hidden_activation in {"sigmoid", "tanh"}:
        saturated_fraction = float(np.mean(np.abs(model.hidden_z) > 5.0))
        if saturated_fraction >= 0.25:
            diagnostics.append(
                Diagnostic(
                    code="activation_saturation",
                    severity="warning",
                    signal=f"{saturated_fraction:.0%} of hidden preactivations exceed |5|.",
                    explanation=(
                        "Saturated activations have small local gradients and may learn slowly."
                    ),
                )
            )
    if loss_history and not np.isfinite(loss_history).all():
        diagnostics.append(
            Diagnostic(
                code="non_finite_loss",
                severity="error",
                signal="Loss contains NaN or infinity.",
                explanation="Training diverged; reduce the learning rate and inspect inputs.",
            )
        )
    if len(loss_history) >= 10 and loss_history[-1] > loss_history[0] * 1.25:
        diagnostics.append(
            Diagnostic(
                code="loss_growth",
                severity="warning",
                signal="Final loss is more than 25% above initial loss.",
                explanation="The update size or initialization may be unstable.",
            )
        )
    if not diagnostics:
        diagnostics.append(
            Diagnostic(
                code="stable",
                severity="success",
                signal="No configured instability signal was detected.",
                explanation="This is a bounded diagnostic, not a guarantee of generalization.",
            )
        )
    return diagnostics
