from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


def regression_metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(target, prediction)),
        "rmse": float(mean_squared_error(target, prediction) ** 0.5),
        "r2": float(r2_score(target, prediction)),
    }


def classification_metrics(
    target: np.ndarray,
    probabilities: np.ndarray,
) -> dict[str, object]:
    prediction = probabilities.argmax(axis=1)
    return {
        "accuracy": float(accuracy_score(target, prediction)),
        "macro_f1": float(f1_score(target, prediction, average="macro")),
        "log_loss": float(log_loss(target, probabilities, labels=np.arange(probabilities.shape[1]))),
        "confusion_matrix": confusion_matrix(target, prediction).tolist(),
    }
