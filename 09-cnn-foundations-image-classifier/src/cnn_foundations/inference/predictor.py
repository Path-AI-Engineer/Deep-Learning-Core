from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import torch

from cnn_foundations.artifacts.bundle import BundleContents
from cnn_foundations.contracts.config import CLASS_NAMES


@dataclass(frozen=True)
class Prediction:
    predicted_index: int
    predicted_class: str
    probabilities: tuple[dict[str, Any], ...]
    top_k: tuple[dict[str, Any], ...]
    model_version: str
    inference_time_ms: float


class Predictor:
    def __init__(self, bundle: BundleContents) -> None:
        self.bundle = bundle
        self.model = bundle.model.cpu().eval()

    def predict(self, tensor: torch.Tensor, *, top_k: int = 3) -> Prediction:
        if tensor.ndim != 4 or tuple(tensor.shape[1:]) != (1, 28, 28):
            raise ValueError("inference requires NCHW input [N, 1, 28, 28].")
        if tensor.shape[0] != 1:
            raise ValueError("interactive inference accepts exactly one image.")
        started = time.perf_counter()
        with torch.inference_mode():
            logits = self.model(tensor.cpu())
            scores = torch.softmax(logits, dim=1)[0]
        elapsed = (time.perf_counter() - started) * 1000
        ranking = torch.argsort(scores, descending=True)
        probability_rows = tuple(
            {
                "index": index,
                "class_name": CLASS_NAMES[index],
                "probability": float(scores[index]),
            }
            for index in range(len(CLASS_NAMES))
        )
        top_rows = tuple(probability_rows[int(index)] for index in ranking[:top_k])
        predicted_index = int(ranking[0])
        return Prediction(
            predicted_index=predicted_index,
            predicted_class=CLASS_NAMES[predicted_index],
            probabilities=probability_rows,
            top_k=top_rows,
            model_version=str(self.bundle.metadata["model_version"]),
            inference_time_ms=elapsed,
        )

