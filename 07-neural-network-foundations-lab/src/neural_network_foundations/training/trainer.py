"""A transparent full-batch training loop for pedagogical datasets."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from neural_network_foundations.evaluation.metrics import binary_accuracy
from neural_network_foundations.models import MLP
from neural_network_foundations.optimizers import SGD

FloatArray = NDArray[np.float64]


@dataclass
class TrainingHistory:
    loss: list[float] = field(default_factory=list)
    accuracy: list[float] = field(default_factory=list)
    status: str = "not_started"
    completed_epochs: int = 0
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def train(
    model: MLP,
    features: FloatArray,
    targets: FloatArray,
    *,
    loss_name: str,
    learning_rate: float,
    epochs: int,
) -> TrainingHistory:
    if not 1 <= epochs <= 5000:
        raise ValueError("epochs must be between 1 and 5000.")
    optimizer = SGD(learning_rate)
    history = TrainingHistory(status="running")
    previous_loss: float | None = None

    for epoch in range(epochs):
        predictions = model.forward(features)
        current_loss = model.calculate_loss(targets, loss_name=loss_name)
        current_accuracy = binary_accuracy(predictions, targets)
        if not np.isfinite(current_loss):
            history.status = "diverged"
            history.warnings.append("Training stopped because loss became non-finite.")
            break
        model.backward(targets, loss_name=loss_name)
        optimizer.step(model.parameters(), model.gradients())
        history.loss.append(current_loss)
        history.accuracy.append(current_accuracy)
        history.completed_epochs = epoch + 1
        if previous_loss is not None and current_loss > previous_loss * 20.0:
            history.warnings.append("Loss increased sharply; review the learning rate.")
        previous_loss = current_loss

    if history.status == "running":
        history.status = "completed"
    model.forward(features)
    return history
