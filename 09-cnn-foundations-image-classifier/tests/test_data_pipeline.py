from __future__ import annotations

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from cnn_foundations.contracts.config import AugmentationConfig, DatasetConfig
from cnn_foundations.data.splits import stratified_train_validation_split
from cnn_foundations.transforms.pipeline import (
    build_inference_transform,
    build_split_transforms,
)


def test_stratified_split_is_reproducible_and_complete() -> None:
    labels = np.repeat(np.arange(10, dtype=np.int64), 20)
    first = stratified_train_validation_split(
        labels, validation_fraction=0.2, seed=42
    )
    second = stratified_train_validation_split(
        labels, validation_fraction=0.2, seed=42
    )
    first.validate(len(labels))
    assert first == second
    assert len(first.validation) == 40


def test_augmentation_is_train_only() -> None:
    train, validation = build_split_transforms(
        DatasetConfig(),
        AugmentationConfig(random_crop_padding=2, horizontal_flip=True),
    )
    assert any(isinstance(step, transforms.RandomCrop) for step in train.transforms)
    assert any(
        isinstance(step, transforms.RandomHorizontalFlip)
        for step in train.transforms
    )
    assert not any(
        isinstance(step, transforms.RandomCrop | transforms.RandomHorizontalFlip)
        for step in validation.transforms
    )


def test_inference_transform_produces_expected_tensor() -> None:
    tensor = build_inference_transform(DatasetConfig())(
        Image.new("RGB", (40, 32), color=(120, 120, 120))
    )
    assert isinstance(tensor, torch.Tensor)
    assert tuple(tensor.shape) == (1, 28, 28)
