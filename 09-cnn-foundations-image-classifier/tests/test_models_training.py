from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from cnn_foundations.models.cnn import FashionCNN
from cnn_foundations.models.mlp import FashionMLP
from cnn_foundations.training.engine import EarlyStopping, run_epoch


def test_cnn_forward_contract() -> None:
    assert FashionCNN()(torch.zeros(4, 1, 28, 28)).shape == (4, 10)


def test_cnn_rejects_non_nchw_input() -> None:
    try:
        FashionCNN()(torch.zeros(4, 28, 28))
    except ValueError as error:
        assert "NCHW" in str(error)
    else:
        raise AssertionError("invalid tensor shape was accepted")


def test_mlp_forward_contract() -> None:
    assert FashionMLP()(torch.zeros(3, 1, 28, 28)).shape == (3, 10)


def test_shape_trace_reaches_ten_logits() -> None:
    assert FashionCNN().shape_trace(batch_size=2)[-1]["shape"] == [2, 10]


def test_training_epoch_updates_weights_and_eval_mode() -> None:
    inputs = torch.rand(8, 1, 28, 28)
    targets = torch.arange(8) % 10
    loader = DataLoader(TensorDataset(inputs, targets), batch_size=4)
    model = FashionMLP(hidden_features=16, dropout=0)
    before = model.network[1].weight.detach().clone()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    result = run_epoch(
        model, loader, nn.CrossEntropyLoss(), device=torch.device("cpu"), optimizer=optimizer
    )
    assert result.examples == 8
    assert not torch.equal(before, model.network[1].weight)
    run_epoch(model, loader, nn.CrossEntropyLoss(), device=torch.device("cpu"))
    assert model.training is False


def test_early_stopping_waits_for_patience() -> None:
    stopper = EarlyStopping(patience=2)
    assert stopper.update(1.0) is False
    assert stopper.update(1.1) is False
    assert stopper.update(1.2) is True
