from __future__ import annotations

import time
import uuid
from dataclasses import dataclass

import numpy as np
import torch
from numpy.typing import NDArray
from torch import nn

from sequence_models.contracts import CLASS_MAPPING


@dataclass(frozen=True, slots=True)
class Prediction:
    prediction_id: str
    predicted_index: int
    predicted_class: str
    probabilities: list[float]
    confidence: float
    latency_ms: float


class SequencePredictor:
    def __init__(
        self,
        model: nn.Module,
        mean: NDArray[np.float32] | None = None,
        std: NDArray[np.float32] | None = None,
    ):
        self._model = model.to("cpu")
        self._model.eval()
        self._mean = mean if mean is not None else np.zeros(9, dtype=np.float32)
        self._std = std if std is not None else np.ones(9, dtype=np.float32)

    def predict(self, values: NDArray[np.float32]) -> Prediction:
        if values.shape != (128, 9):
            raise ValueError("sequence must have shape [128, 9]")
        if not np.isfinite(values).all():
            raise ValueError("sequence contains non-finite values")
        normalized = ((values - self._mean) / self._std).astype(np.float32)
        inputs = torch.from_numpy(normalized).unsqueeze(0)
        started = time.perf_counter()
        with torch.inference_mode():
            logits = self._model(inputs)
            probabilities_tensor = torch.softmax(logits, dim=1)[0]
        latency_ms = (time.perf_counter() - started) * 1000
        probabilities = [float(value) for value in probabilities_tensor.tolist()]
        predicted_index = int(np.argmax(probabilities))
        return Prediction(
            prediction_id=uuid.uuid4().hex,
            predicted_index=predicted_index,
            predicted_class=CLASS_MAPPING[predicted_index],
            probabilities=probabilities,
            confidence=probabilities[predicted_index],
            latency_ms=latency_ms,
        )
