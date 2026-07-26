from __future__ import annotations

import torch
from torch import nn

from pytorch_tabular.models import ClassificationMLP, RegressionMLP


def test_regression_model_shape_and_gradients() -> None:
    model = RegressionMLP(input_features=8)
    features = torch.randn(5, 8)
    target = torch.randn(5)
    prediction = model(features)
    nn.MSELoss()(prediction, target).backward()
    assert prediction.shape == (5,)
    assert all(parameter.grad is not None for parameter in model.parameters())


def test_classifier_emits_logits_and_valid_probabilities() -> None:
    model = ClassificationMLP(input_features=13, class_count=3)
    logits = model(torch.randn(7, 13))
    probabilities = torch.softmax(logits, dim=1)
    assert logits.shape == (7, 3)
    assert torch.allclose(probabilities.sum(dim=1), torch.ones(7))


def test_cross_entropy_updates_parameters_without_pre_softmax() -> None:
    model = ClassificationMLP(input_features=13, class_count=3)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    before = [parameter.detach().clone() for parameter in model.parameters()]
    loss = nn.CrossEntropyLoss()(model(torch.randn(8, 13)), torch.arange(8) % 3)
    loss.backward()
    optimizer.step()
    assert any(not torch.equal(old, new) for old, new in zip(before, model.parameters()))
