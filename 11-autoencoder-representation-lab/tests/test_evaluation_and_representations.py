from __future__ import annotations

import numpy as np
import pytest
import torch

from autoencoder_lab.evaluation import per_sample_mse, reconstruction_metrics
from autoencoder_lab.representations import interpolate, nearest_neighbors


def test_perfect_reconstruction_metrics() -> None:
    values = torch.rand(2, 1, 28, 28)
    metrics = reconstruction_metrics(values, values)
    assert metrics.mse == 0
    assert metrics.mae == 0
    assert metrics.psnr is None
    assert metrics.ssim == 1


def test_metrics_reject_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="shapes must match"):
        reconstruction_metrics(torch.zeros(1, 1, 28, 28), torch.zeros(2, 1, 28, 28))


def test_per_sample_mse_keeps_samples_separate() -> None:
    target = torch.zeros(2, 1, 28, 28)
    prediction = target.clone()
    prediction[1] = 1
    assert per_sample_mse(target, prediction).tolist() == [0.0, 1.0]


def test_interpolation_has_exact_endpoints() -> None:
    start = torch.tensor([[0.0, 1.0]])
    end = torch.tensor([[1.0, 0.0]])
    path, alphas = interpolate(start, end, 5)
    assert torch.equal(path[0], start[0])
    assert torch.equal(path[-1], end[0])
    assert alphas == [0.0, 0.25, 0.5, 0.75, 1.0]


def test_interpolation_rejects_too_few_steps() -> None:
    with pytest.raises(ValueError, match="between 3 and 12"):
        interpolate(torch.zeros(1, 2), torch.ones(1, 2), 2)


def test_nearest_neighbors_excludes_query() -> None:
    embeddings = np.asarray([[0, 0], [1, 0], [2, 0]], dtype=np.float32)
    rows = nearest_neighbors(
        np.asarray([0, 0], dtype=np.float32),
        embeddings,
        ["a", "b", "c"],
        limit=2,
        exclude_id="a",
    )
    assert [row["sample_id"] for row in rows] == ["b", "c"]
