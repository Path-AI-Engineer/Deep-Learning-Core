from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from numpy.typing import NDArray
from sklearn.model_selection import train_test_split


@dataclass(frozen=True)
class SplitIndices:
    train: tuple[int, ...]
    validation: tuple[int, ...]
    seed: int
    validation_fraction: float

    def as_dict(self) -> dict[str, object]:
        return asdict(self)

    def validate(self, source_size: int) -> None:
        train_set = set(self.train)
        validation_set = set(self.validation)
        if train_set & validation_set:
            raise ValueError("train and validation indices overlap.")
        if train_set | validation_set != set(range(source_size)):
            raise ValueError("split indices do not cover the training source exactly.")


def stratified_train_validation_split(
    labels: NDArray[np.int64],
    *,
    validation_fraction: float,
    seed: int,
) -> SplitIndices:
    targets = np.asarray(labels, dtype=np.int64)
    if targets.ndim != 1 or targets.size < 20:
        raise ValueError("labels must be a one-dimensional classification target.")
    indices = np.arange(targets.size)
    train, validation = train_test_split(
        indices,
        test_size=validation_fraction,
        random_state=seed,
        shuffle=True,
        stratify=targets,
    )
    result = SplitIndices(
        train=tuple(sorted(int(value) for value in train)),
        validation=tuple(sorted(int(value) for value in validation)),
        seed=seed,
        validation_fraction=validation_fraction,
    )
    result.validate(targets.size)
    return result
