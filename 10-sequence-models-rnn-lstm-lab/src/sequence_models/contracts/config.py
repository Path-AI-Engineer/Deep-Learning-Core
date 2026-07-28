from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

ModelType = Literal["statistics-mlp", "rnn", "lstm", "gru"]

CHANNELS = (
    "body_acc_x",
    "body_acc_y",
    "body_acc_z",
    "body_gyro_x",
    "body_gyro_y",
    "body_gyro_z",
    "total_acc_x",
    "total_acc_y",
    "total_acc_z",
)

CLASS_MAPPING = {
    0: "WALKING",
    1: "WALKING_UPSTAIRS",
    2: "WALKING_DOWNSTAIRS",
    3: "SITTING",
    4: "STANDING",
    5: "LAYING",
}

CLASS_DESCRIPTIONS = {
    0: "Level walking captured from waist-mounted inertial sensors.",
    1: "Ascending stairs with periodic vertical acceleration.",
    2: "Descending stairs with sharper impact transitions.",
    3: "Seated posture with low dynamic movement.",
    4: "Upright stationary posture.",
    5: "Horizontal stationary posture.",
}


@dataclass(frozen=True, slots=True)
class ModelConfig:
    model_type: ModelType
    input_size: int = 9
    hidden_size: int = 24
    num_layers: int = 1
    num_classes: int = 6
    dropout: float = 0.15
    batch_first: bool = True

    def validate(self) -> None:
        if self.input_size != len(CHANNELS):
            raise ValueError("input_size must match the nine UCI HAR channels")
        if self.num_classes != len(CLASS_MAPPING):
            raise ValueError("num_classes must match the six activities")
        if self.num_layers < 1 or self.hidden_size < 1:
            raise ValueError("hidden_size and num_layers must be positive")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    seed: int = 42
    epochs: int = 18
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    gradient_clip_norm: float = 1.0
    early_stopping_patience: int = 4

    def validate(self) -> None:
        if min(self.epochs, self.batch_size, self.early_stopping_patience) < 1:
            raise ValueError("epochs, batch_size and patience must be positive")
        if self.learning_rate <= 0 or self.gradient_clip_norm <= 0:
            raise ValueError("learning_rate and clipping threshold must be positive")
