from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)

from sequence_models.contracts import CLASS_MAPPING


@dataclass(frozen=True, slots=True)
class ClassificationMetrics:
    accuracy: float
    macro_precision: float
    macro_recall: float
    macro_f1: float
    per_class: list[dict[str, float | int | str]]
    confusion_matrix: list[list[int]]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def classification_metrics(
    true_labels: NDArray[np.int64],
    predicted_labels: NDArray[np.int64],
) -> ClassificationMetrics:
    labels = list(CLASS_MAPPING)
    precision, recall, f1, support = precision_recall_fscore_support(
        true_labels,
        predicted_labels,
        labels=labels,
        zero_division=0,
    )
    macro = precision_recall_fscore_support(
        true_labels,
        predicted_labels,
        labels=labels,
        average="macro",
        zero_division=0,
    )
    per_class: list[dict[str, float | int | str]] = []
    for index, label in enumerate(labels):
        per_class.append(
            {
                "class_index": label,
                "class_name": CLASS_MAPPING[label],
                "precision": round(float(precision[index]), 6),
                "recall": round(float(recall[index]), 6),
                "f1": round(float(f1[index]), 6),
                "support": int(support[index]),
            }
        )
    return ClassificationMetrics(
        accuracy=round(float(accuracy_score(true_labels, predicted_labels)), 6),
        macro_precision=round(float(macro[0]), 6),
        macro_recall=round(float(macro[1]), 6),
        macro_f1=round(float(macro[2]), 6),
        per_class=per_class,
        confusion_matrix=confusion_matrix(true_labels, predicted_labels, labels=labels).tolist(),
    )
