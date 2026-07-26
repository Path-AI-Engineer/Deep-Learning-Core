from __future__ import annotations

from pytorch_tabular.contracts import ExperimentConfig
from pytorch_tabular.experiments import run_experiment


def test_regressor_beats_mean_baseline_on_mae(runtime_dir) -> None:
    result = run_experiment(
        ExperimentConfig(
            task="regression",
            epochs=120,
            patience=14,
            batch_size=128,
        ),
        runtime_dir,
    )
    assert result.metrics["mae"] < result.baseline_metrics["mae"]
