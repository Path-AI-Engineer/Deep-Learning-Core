from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

from cnn_foundations.contracts.config import CLASS_NAMES


@dataclass(frozen=True)
class ClassificationReport:
    accuracy: float
    macro_f1: float
    per_class: tuple[dict[str, Any], ...]
    confusion_matrix: tuple[tuple[int, ...], ...]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def evaluate_predictions(
    expected: NDArray[np.int64] | list[int],
    predicted: NDArray[np.int64] | list[int],
) -> ClassificationReport:
    y_true = np.asarray(expected, dtype=np.int64)
    y_pred = np.asarray(predicted, dtype=np.int64)
    if y_true.shape != y_pred.shape or y_true.ndim != 1:
        raise ValueError("expected and predicted labels must have the same 1D shape.")
    labels = np.arange(len(CLASS_NAMES))
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        zero_division=0,
    )
    per_class = tuple(
        {
            "index": index,
            "class_name": CLASS_NAMES[index],
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(f1[index]),
            "support": int(support[index]),
        }
        for index in labels
    )
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    return ClassificationReport(
        accuracy=float(accuracy_score(y_true, y_pred)),
        macro_f1=float(
            f1_score(
                y_true,
                y_pred,
                labels=labels,
                average="macro",
                zero_division=0,
            )
        ),
        per_class=per_class,
        confusion_matrix=tuple(tuple(int(value) for value in row) for row in matrix),
    )
