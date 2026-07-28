import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from sequence_models.contracts import ModelConfig, TrainingConfig
from sequence_models.evaluation import classification_metrics
from sequence_models.experiments import gradient_flow_experiment, permute_timesteps
from sequence_models.models import build_model
from sequence_models.training import train


def test_classification_metrics_include_every_class() -> None:
    truth = np.arange(6, dtype=np.int64)
    metrics = classification_metrics(truth, truth)
    assert metrics.accuracy == 1.0
    assert metrics.macro_f1 == 1.0
    assert len(metrics.per_class) == 6
    assert len(metrics.confusion_matrix) == 6


def test_temporal_permutation_is_deterministic_and_preserves_shape() -> None:
    inputs = torch.arange(2 * 12 * 9).reshape(2, 12, 9)
    first = permute_timesteps(inputs, 42)
    second = permute_timesteps(inputs, 42)
    assert torch.equal(first, second)
    assert first.shape == inputs.shape
    assert not torch.equal(first, inputs)


def test_gradient_experiment_is_finite_and_reproducible() -> None:
    first = gradient_flow_experiment()
    second = gradient_flow_experiment()
    assert first == second
    assert first["finite"] is True
    assert len(first["scenarios"]) == 3


def test_trainer_records_clipping_and_selects_checkpoint() -> None:
    torch.manual_seed(42)
    values = torch.randn(18, 12, 9)
    labels = torch.arange(18) % 6
    loader = DataLoader(TensorDataset(values, labels), batch_size=6, shuffle=False)
    model = build_model(ModelConfig(model_type="rnn", hidden_size=8, dropout=0.0))
    result = train(
        model,
        loader,
        loader,
        TrainingConfig(epochs=2, batch_size=6, early_stopping_patience=2),
    )
    assert result.best_epoch >= 1
    assert result.state_dict
    assert all(
        epoch.gradient_norm_after_clip <= epoch.gradient_norm_before_clip + 1e-7
        for epoch in result.history
    )
