from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from sklearn.model_selection import train_test_split
from torchvision.datasets import FashionMNIST


@dataclass(frozen=True, slots=True)
class PreparedFashionMNIST:
    train_images: NDArray[np.float32]
    train_labels: NDArray[np.int64]
    validation_images: NDArray[np.float32]
    validation_labels: NDArray[np.int64]
    test_images: NDArray[np.float32]
    test_labels: NDArray[np.int64]


def _images(dataset: FashionMNIST) -> NDArray[np.float32]:
    values = dataset.data.numpy().astype(np.float32) / 255.0
    return np.asarray(values[:, None, :, :], dtype=np.float32)


def _labels(dataset: FashionMNIST) -> NDArray[np.int64]:
    return np.asarray(dataset.targets, dtype=np.int64)


def prepare_fashion_mnist(root: Path, download: bool = False) -> PreparedFashionMNIST:
    official_train = FashionMNIST(root=root, train=True, download=download)
    official_test = FashionMNIST(root=root, train=False, download=download)
    images, labels = _images(official_train), _labels(official_train)
    train_index, validation_index = train_test_split(
        np.arange(len(images)),
        test_size=0.1,
        random_state=42,
        stratify=labels,
    )
    prepared = PreparedFashionMNIST(
        train_images=images[train_index],
        train_labels=labels[train_index],
        validation_images=images[validation_index],
        validation_labels=labels[validation_index],
        test_images=_images(official_test),
        test_labels=_labels(official_test),
    )
    for values in (
        prepared.train_images,
        prepared.validation_images,
        prepared.test_images,
    ):
        if values.dtype != np.float32 or values.shape[1:] != (1, 28, 28):
            raise ValueError("FashionMNIST tensor contract failed")
        if not np.isfinite(values).all() or values.min() < 0 or values.max() > 1:
            raise ValueError("FashionMNIST pixels must be finite and in [0, 1]")
    return prepared


def checksum_files(root: Path) -> dict[str, str]:
    checksums: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if path.is_file():
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            checksums[str(path.relative_to(root)).replace("\\", "/")] = digest
    return checksums


def write_split_manifest(path: Path, prepared: PreparedFashionMNIST) -> None:
    payload = {
        "strategy": "official test; deterministic stratified 90/10 official-train split",
        "seed": 42,
        "labels_used_for_autoencoder_training": False,
        "sizes": {
            "train": len(prepared.train_images),
            "validation": len(prepared.validation_images),
            "test": len(prepared.test_images),
        },
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
