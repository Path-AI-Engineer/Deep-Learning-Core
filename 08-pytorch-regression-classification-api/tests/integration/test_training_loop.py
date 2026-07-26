from __future__ import annotations

import copy

import torch

from pytorch_tabular.contracts import ExperimentConfig
from pytorch_tabular.data import load_task_data
from pytorch_tabular.models import ClassificationMLP
from pytorch_tabular.training import fit, validate_epoch


def test_classification_training_reduces_validation_loss() -> None:
    data = load_task_data("classification")
    loaders = data.loaders(batch_size=24)
    model = ClassificationMLP(13, 3, dropout=0.0)
    result = fit(
        model,
        loaders.train,
        loaders.validation,
        ExperimentConfig(task="classification", epochs=50, patience=12),
    )
    assert result.history.validation_loss[-1] < result.history.validation_loss[0]


def test_validation_does_not_update_parameters() -> None:
    data = load_task_data("classification")
    model = ClassificationMLP(13, 3)
    before = copy.deepcopy(model.state_dict())
    validate_epoch(
        model,
        data.loaders(24).validation,
        torch.nn.CrossEntropyLoss(),
        torch.device("cpu"),
    )
    assert all(torch.equal(before[key], value) for key, value in model.state_dict().items())
