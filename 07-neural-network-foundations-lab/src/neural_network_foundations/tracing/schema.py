"""Versioned public trace objects consumed by the visual explainer."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any

import numpy as np


def _serializable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.floating | np.integer):
        return value.item()
    if isinstance(value, dict):
        return {key: _serializable(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_serializable(item) for item in value]
    return value


@dataclass
class TraceNode:
    layer_id: str
    neuron_id: str
    inputs: list[float]
    weights: list[float]
    bias: float
    z: float
    activation_name: str
    activation_value: float
    upstream_gradient: float | None = None
    local_gradient: float | None = None
    parameter_gradients: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return _serializable(asdict(self))


@dataclass
class ExecutionTrace:
    dataset: str
    sample_index: int
    target: float
    prediction: float
    loss_name: str
    loss: float
    nodes: list[TraceNode]
    configuration: dict[str, Any]
    schema_version: str = "1.0"
    generated_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    def to_dict(self) -> dict[str, Any]:
        return _serializable(asdict(self))
