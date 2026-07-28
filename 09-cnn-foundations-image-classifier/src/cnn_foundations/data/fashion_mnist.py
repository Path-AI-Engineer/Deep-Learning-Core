from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.datasets import FashionMNIST

from cnn_foundations.contracts.config import AugmentationConfig, DatasetConfig
from cnn_foundations.data.splits import SplitIndices, stratified_train_validation_split
from cnn_foundations.transforms.pipeline import build_split_transforms


@dataclass(frozen=True)
class FashionDataLoaders:
    train: DataLoader[tuple[torch.Tensor, torch.Tensor]]
    validation: DataLoader[tuple[torch.Tensor, torch.Tensor]]
    test: DataLoader[tuple[torch.Tensor, torch.Tensor]]
    split: SplitIndices


def load_fashion_mnist(
    config: DatasetConfig,
    *,
    train: bool,
    transform: object | None = None,
    download: bool = False,
) -> FashionMNIST:
    root = Path(config.root)
    try:
        return FashionMNIST(
            root=root,
            train=train,
            transform=transform,
            download=download,
        )
    except RuntimeError as error:
        raise RuntimeError(
            "FashionMNIST is unavailable. Run scripts/prepare_data.py from a "
            "network-enabled terminal before training."
        ) from error


def build_dataloaders(
    dataset_config: DatasetConfig,
    augmentation: AugmentationConfig,
    *,
    batch_size: int,
    num_workers: int = 0,
    download: bool = False,
) -> FashionDataLoaders:
    train_transform, deterministic_transform = build_split_transforms(
        dataset_config, augmentation
    )
    source_for_train = load_fashion_mnist(
        dataset_config, train=True, transform=train_transform, download=download
    )
    source_for_validation = load_fashion_mnist(
        dataset_config, train=True, transform=deterministic_transform, download=False
    )
    test_dataset = load_fashion_mnist(
        dataset_config, train=False, transform=deterministic_transform, download=download
    )
    labels = np.asarray(source_for_train.targets, dtype=np.int64)
    split = stratified_train_validation_split(
        labels,
        validation_fraction=dataset_config.validation_fraction,
        seed=dataset_config.seed,
    )
    generator = torch.Generator().manual_seed(dataset_config.seed)
    train_subset: Dataset[tuple[torch.Tensor, int]] = Subset(
        source_for_train, split.train
    )
    validation_subset: Dataset[tuple[torch.Tensor, int]] = Subset(
        source_for_validation, split.validation
    )
    return FashionDataLoaders(
        train=cast(
            DataLoader[tuple[torch.Tensor, torch.Tensor]],
            DataLoader(
                train_subset,
                batch_size=batch_size,
                shuffle=True,
                generator=generator,
                num_workers=num_workers,
            ),
        ),
        validation=cast(
            DataLoader[tuple[torch.Tensor, torch.Tensor]],
            DataLoader(
                validation_subset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
            ),
        ),
        test=cast(
            DataLoader[tuple[torch.Tensor, torch.Tensor]],
            DataLoader(
                test_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
            ),
        ),
        split=split,
    )
