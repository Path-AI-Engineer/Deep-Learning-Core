from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from sequence_models.contracts import CHANNELS

_UCI_SIGNAL_FILES = tuple(name.replace("_", "_") for name in CHANNELS)


@dataclass(frozen=True, slots=True)
class RawSequences:
    values: NDArray[np.float32]
    labels: NDArray[np.int64]
    subjects: NDArray[np.int64]


@dataclass(frozen=True, slots=True)
class PreparedSequences:
    train: RawSequences
    validation: RawSequences
    test: RawSequences
    mean: NDArray[np.float32]
    std: NDArray[np.float32]


def _resolve_dataset_root(path: Path) -> Path:
    candidates = (path, path / "UCI HAR Dataset")
    for candidate in candidates:
        if (candidate / "train" / "Inertial Signals").is_dir():
            return candidate
    raise FileNotFoundError(
        "UCI HAR extraction was not found. Expected 'UCI HAR Dataset/train/Inertial Signals'."
    )


def _load_split(root: Path, split: str) -> RawSequences:
    split_root = root / split
    signal_root = split_root / "Inertial Signals"
    signal_matrices: list[NDArray[np.float32]] = []
    for channel in _UCI_SIGNAL_FILES:
        file_path = signal_root / f"{channel}_{split}.txt"
        if not file_path.is_file():
            raise FileNotFoundError(f"Missing UCI HAR signal file: {file_path.name}")
        matrix = np.loadtxt(file_path, dtype=np.float32)
        if matrix.ndim != 2 or matrix.shape[1] != 128:
            raise ValueError(f"{file_path.name} must have shape [N, 128]")
        signal_matrices.append(matrix)
    values = np.stack(signal_matrices, axis=2).astype(np.float32)
    labels = np.loadtxt(split_root / f"y_{split}.txt", dtype=np.int64).reshape(-1) - 1
    subjects = np.loadtxt(split_root / f"subject_{split}.txt", dtype=np.int64).reshape(-1)
    if not (len(values) == len(labels) == len(subjects)):
        raise ValueError(f"UCI HAR {split} arrays do not contain the same number of rows")
    if not np.isfinite(values).all():
        raise ValueError(f"UCI HAR {split} contains NaN or infinite values")
    if labels.min() < 0 or labels.max() > 5:
        raise ValueError("UCI HAR labels must map to the inclusive range 0..5")
    return RawSequences(values=values, labels=labels, subjects=subjects)


def load_uci_har(path: str | Path) -> tuple[RawSequences, RawSequences]:
    root = _resolve_dataset_root(Path(path))
    return _load_split(root, "train"), _load_split(root, "test")


def _slice(data: RawSequences, mask: NDArray[np.bool_]) -> RawSequences:
    return RawSequences(data.values[mask], data.labels[mask], data.subjects[mask])


def prepare_grouped_splits(
    official_train: RawSequences,
    official_test: RawSequences,
    validation_subjects: tuple[int, ...] = (1, 3, 5, 7),
) -> PreparedSequences:
    validation_mask = np.isin(official_train.subjects, validation_subjects)
    training_mask = ~validation_mask
    if not validation_mask.any() or not training_mask.any():
        raise ValueError("validation_subjects must create two non-empty grouped splits")
    train = _slice(official_train, training_mask)
    validation = _slice(official_train, validation_mask)
    if set(train.subjects.tolist()) & set(validation.subjects.tolist()):
        raise ValueError("subject leakage detected between training and validation")
    if set(official_test.subjects.tolist()) & set(official_train.subjects.tolist()):
        raise ValueError("official test subjects overlap with official training subjects")
    mean = train.values.mean(axis=(0, 1), dtype=np.float64).astype(np.float32)
    std = train.values.std(axis=(0, 1), dtype=np.float64).astype(np.float32)
    std = np.where(std < 1e-6, np.float32(1.0), std).astype(np.float32)

    def normalize(data: RawSequences) -> RawSequences:
        normalized = ((data.values - mean) / std).astype(np.float32)
        return RawSequences(normalized, data.labels.copy(), data.subjects.copy())

    return PreparedSequences(
        train=normalize(train),
        validation=normalize(validation),
        test=normalize(official_test),
        mean=mean,
        std=std,
    )
