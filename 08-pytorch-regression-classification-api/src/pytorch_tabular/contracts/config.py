from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

TaskName = Literal["regression", "classification"]


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    display_name: str
    description: str
    minimum: float
    maximum: float
    example: float
    unit: str | None = None


@dataclass(frozen=True)
class ExperimentConfig:
    task: TaskName
    seed: int = 42
    batch_size: int = 32
    epochs: int = 120
    learning_rate: float = 0.001
    optimizer: Literal["adam", "sgd"] = "adam"
    weight_decay: float = 0.0001
    patience: int = 14
    min_delta: float = 0.0001

    def to_dict(self) -> dict[str, object]:
        return asdict(self)
