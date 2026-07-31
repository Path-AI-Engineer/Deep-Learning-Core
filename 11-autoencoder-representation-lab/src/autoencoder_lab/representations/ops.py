from __future__ import annotations

from typing import cast

import numpy as np
import torch
from numpy.typing import NDArray
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch import Tensor, nn

from autoencoder_lab.protocols import EncoderDecoder


def extract_embeddings(model: nn.Module, images: Tensor) -> NDArray[np.float32]:
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    with torch.inference_mode():
        encoded = cast(EncoderDecoder, model).encode(images)
    return np.asarray(encoded.detach().cpu().numpy(), dtype=np.float32)


def train_linear_probe(
    train_embeddings: NDArray[np.float32],
    train_labels: NDArray[np.int64],
    validation_embeddings: NDArray[np.float32],
    validation_labels: NDArray[np.int64],
) -> dict[str, float]:
    probe = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, random_state=42),
    )
    probe.fit(train_embeddings, train_labels)
    prediction = probe.predict(validation_embeddings)
    return {
        "accuracy": round(float(accuracy_score(validation_labels, prediction)), 6),
        "macro_f1": round(
            float(f1_score(validation_labels, prediction, average="macro", zero_division=0)),
            6,
        ),
    }


def nearest_neighbors(
    query: NDArray[np.float32],
    embeddings: NDArray[np.float32],
    sample_ids: list[str],
    limit: int = 5,
    exclude_id: str | None = None,
) -> list[dict[str, float | str]]:
    if not 1 <= limit <= 8:
        raise ValueError("neighbor limit must be between 1 and 8")
    distances = np.linalg.norm(embeddings - query.reshape(1, -1), axis=1)
    order = np.argsort(distances)
    result: list[dict[str, float | str]] = []
    for index in order:
        if sample_ids[index] == exclude_id:
            continue
        result.append(
            {"sample_id": sample_ids[index], "distance": round(float(distances[index]), 6)}
        )
        if len(result) == limit:
            break
    return result


def interpolate(start: Tensor, end: Tensor, steps: int) -> tuple[Tensor, list[float]]:
    if start.shape != end.shape or start.ndim != 2:
        raise ValueError("latent endpoints must be matching [N, latent_dim] tensors")
    if not 3 <= steps <= 12:
        raise ValueError("interpolation steps must be between 3 and 12")
    alphas = torch.linspace(0, 1, steps, dtype=start.dtype, device=start.device)
    path = torch.stack([(1 - alpha) * start[0] + alpha * end[0] for alpha in alphas])
    return path, [round(float(value), 4) for value in alphas]
