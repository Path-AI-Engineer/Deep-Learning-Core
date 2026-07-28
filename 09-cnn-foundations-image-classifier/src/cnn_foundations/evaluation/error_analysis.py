from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from cnn_foundations.contracts.config import CLASS_NAMES


def build_error_records(
    expected: list[int],
    probabilities: NDArray[np.float64],
    sample_ids: list[str],
    *,
    limit: int = 100,
) -> list[dict[str, Any]]:
    if probabilities.ndim != 2 or probabilities.shape[1] != len(CLASS_NAMES):
        raise ValueError("probabilities must have shape [N, 10].")
    if not (len(expected) == probabilities.shape[0] == len(sample_ids)):
        raise ValueError("error-analysis inputs must share the same sample count.")
    predicted = probabilities.argmax(axis=1)
    records: list[dict[str, Any]] = []
    for index, (truth, prediction) in enumerate(zip(expected, predicted, strict=True)):
        if int(truth) == int(prediction):
            continue
        records.append(
            {
                "sample_id": sample_ids[index],
                "true_index": int(truth),
                "true_class": CLASS_NAMES[int(truth)],
                "predicted_index": int(prediction),
                "predicted_class": CLASS_NAMES[int(prediction)],
                "confidence": float(probabilities[index, prediction]),
            }
        )
        if len(records) >= limit:
            break
    return records
