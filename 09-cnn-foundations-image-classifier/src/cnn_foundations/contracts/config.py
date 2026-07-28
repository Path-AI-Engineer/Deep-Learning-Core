from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

CLASS_NAMES = (
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class DatasetConfig(StrictModel):
    name: Literal["FashionMNIST"] = "FashionMNIST"
    root: Path = Path("data/raw")
    seed: int = 20260728
    validation_fraction: float = Field(default=0.15, gt=0.0, lt=0.5)
    mean: float = Field(default=0.2860406, ge=0.0, le=1.0)
    std: float = Field(default=0.35302424, gt=0.0, le=1.0)
    input_shape: tuple[int, int, int] = (1, 28, 28)
    class_names: tuple[str, ...] = CLASS_NAMES

    @model_validator(mode="after")
    def validate_contract(self) -> DatasetConfig:
        if self.input_shape != (1, 28, 28):
            raise ValueError("FashionMNIST input_shape must be [1, 28, 28].")
        if self.class_names != CLASS_NAMES:
            raise ValueError("FashionMNIST class mapping must remain stable.")
        return self


class ModelConfig(StrictModel):
    kind: Literal["cnn", "mlp"]
    num_classes: int = Field(default=10, ge=2, le=100)
    hidden_features: int = Field(default=128, ge=8, le=1024)
    dropout: float = Field(default=0.25, ge=0.0, lt=0.8)
    input_channels: int | None = None
    input_features: int | None = None
    conv_channels: tuple[int, int] | None = None
    batch_norm: bool = False

    @model_validator(mode="after")
    def validate_architecture(self) -> ModelConfig:
        if self.num_classes != 10:
            raise ValueError("FashionMNIST models must produce ten logits.")
        if self.kind == "cnn":
            if self.input_channels != 1 or not self.conv_channels:
                raise ValueError("CNN requires one input channel and two convolution widths.")
        if self.kind == "mlp" and self.input_features != 784:
            raise ValueError("MLP baseline requires 784 flattened input features.")
        return self


class AugmentationConfig(StrictModel):
    random_crop_padding: int = Field(default=0, ge=0, le=4)
    horizontal_flip: bool = False


class ExperimentConfig(StrictModel):
    name: str = Field(min_length=3, max_length=80)
    model: Path
    dataset: Path
    seed: int
    epochs: int = Field(ge=1, le=50)
    batch_size: int = Field(ge=8, le=512)
    learning_rate: float = Field(gt=0.0, le=0.1)
    weight_decay: float = Field(ge=0.0, le=0.1)
    patience: int = Field(ge=1, le=20)
    device: Literal["cpu", "cuda", "auto"] = "cpu"
    augmentation: AugmentationConfig = AugmentationConfig()


def _load_yaml(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a YAML object.")
    return payload


def load_experiment_config(path: str | Path) -> tuple[ExperimentConfig, DatasetConfig, ModelConfig]:
    experiment_path = Path(path)
    experiment = ExperimentConfig.model_validate(_load_yaml(experiment_path))
    dataset = DatasetConfig.model_validate(_load_yaml(experiment.dataset))
    model = ModelConfig.model_validate(_load_yaml(experiment.model))
    return experiment, dataset, model

