from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

ModelType = Literal["dense-ae", "conv-ae", "denoising-ae", "latent-2d"]
CorruptionType = Literal["gaussian", "masking"]

CLASS_MAPPING = {
    0: "T-shirt/top",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle boot",
}


@dataclass(frozen=True, slots=True)
class ModelConfig:
    model_type: ModelType
    latent_dim: int = 16
    input_shape: tuple[int, int, int] = (1, 28, 28)

    def validate(self) -> None:
        if self.latent_dim not in (2, 8, 16, 32):
            raise ValueError("latent_dim must be one of 2, 8, 16 or 32")
        if self.input_shape != (1, 28, 28):
            raise ValueError("input_shape must be [1, 28, 28]")
        if self.model_type == "latent-2d" and self.latent_dim != 2:
            raise ValueError("latent-2d requires latent_dim=2")


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    seed: int = 42
    epochs: int = 12
    batch_size: int = 32
    learning_rate: float = 3e-3
    weight_decay: float = 1e-4
    early_stopping_patience: int = 4

    def validate(self) -> None:
        if min(self.epochs, self.batch_size, self.early_stopping_patience) < 1:
            raise ValueError("epochs, batch_size and patience must be positive")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("optimizer values are invalid")
