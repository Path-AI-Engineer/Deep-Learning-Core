"""Validated, serializable experiment configuration."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

ACTIVATIONS = {"sigmoid", "tanh", "relu"}
INITIALIZATIONS = {"zeros", "small_normal", "xavier", "he"}
LOSSES = {"mean_squared_error", "binary_cross_entropy"}
DATASETS = {"and", "or", "xor", "circles"}


class ValidationError(ValueError):
    """Raised when a public engine configuration violates safe limits."""


@dataclass(frozen=True)
class NetworkConfig:
    input_features: int = 2
    hidden_units: int = 4
    output_units: int = 1
    hidden_activation: str = "tanh"
    output_activation: str = "sigmoid"
    initialization: str = "xavier"

    def __post_init__(self) -> None:
        if self.input_features != 2:
            raise ValidationError("The visual lab requires exactly two input features.")
        if not 2 <= self.hidden_units <= 8:
            raise ValidationError("hidden_units must be between 2 and 8.")
        if self.output_units != 1:
            raise ValidationError("Binary experiments require exactly one output unit.")
        if self.hidden_activation not in ACTIVATIONS:
            raise ValidationError(f"Unsupported hidden activation: {self.hidden_activation}.")
        if self.output_activation != "sigmoid":
            raise ValidationError("Binary experiments require sigmoid output activation.")
        if self.initialization not in INITIALIZATIONS:
            raise ValidationError(f"Unsupported initialization: {self.initialization}.")

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> NetworkConfig:
        return cls(**value)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ExperimentConfig:
    dataset: str = "xor"
    network: NetworkConfig = field(default_factory=NetworkConfig)
    loss: str = "binary_cross_entropy"
    learning_rate: float = 0.5
    epochs: int = 3000
    seed: int = 7
    grid_resolution: int = 40

    def __post_init__(self) -> None:
        if self.dataset not in DATASETS:
            raise ValidationError(f"Unsupported dataset: {self.dataset}.")
        if self.loss not in LOSSES:
            raise ValidationError(f"Unsupported loss: {self.loss}.")
        if not 0.0001 <= self.learning_rate <= 1.0:
            raise ValidationError("learning_rate must be between 0.0001 and 1.0.")
        if not 1 <= self.epochs <= 5000:
            raise ValidationError("epochs must be between 1 and 5000.")
        if not 0 <= self.seed <= 2_147_483_647:
            raise ValidationError("seed must be a non-negative 32-bit integer.")
        if not 10 <= self.grid_resolution <= 100:
            raise ValidationError("grid_resolution must be between 10 and 100.")
        if self.loss == "binary_cross_entropy" and self.network.output_activation != "sigmoid":
            raise ValidationError("Binary Cross-Entropy requires sigmoid output.")

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> ExperimentConfig:
        raw = dict(value)
        network = raw.get("network", {})
        raw["network"] = (
            network if isinstance(network, NetworkConfig) else NetworkConfig.from_dict(network)
        )
        return cls(**raw)

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["schema_version"] = "1.0"
        return value
