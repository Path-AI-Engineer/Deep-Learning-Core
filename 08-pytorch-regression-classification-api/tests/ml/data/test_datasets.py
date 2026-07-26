from __future__ import annotations

import numpy as np
import torch

from pytorch_tabular.data import load_task_data


def test_wine_split_is_disjoint_and_stratified() -> None:
    data = load_task_data("classification", seed=42)
    assert data.split_summary() == {"train": 124, "validation": 27, "test": 27}
    assert set(np.unique(data.y_train)) == {0, 1, 2}
    assert len(data.feature_names) == 13


def test_scaler_is_fit_on_training_data_only() -> None:
    data = load_task_data("classification", seed=42)
    assert np.allclose(data.x_train.mean(axis=0), 0.0, atol=1e-5)
    assert not np.allclose(data.x_test.mean(axis=0), 0.0, atol=1e-2)


def test_dataloader_shapes_and_target_dtype() -> None:
    data = load_task_data("classification", seed=42)
    features, target = next(iter(data.loaders(batch_size=16).train))
    assert tuple(features.shape) == (16, 13)
    assert target.dtype == torch.int64
