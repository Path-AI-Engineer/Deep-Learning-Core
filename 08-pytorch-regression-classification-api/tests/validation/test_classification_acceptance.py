from __future__ import annotations

from pytorch_tabular.contracts import ExperimentConfig
from pytorch_tabular.experiments import run_experiment


def test_classifier_beats_prior_baseline_on_macro_f1(runtime_dir) -> None:
    result = run_experiment(
        ExperimentConfig(
            task="classification",
            epochs=120,
            patience=18,
            batch_size=24,
        ),
        runtime_dir,
    )
    assert result.metrics["macro_f1"] > result.baseline_metrics["macro_f1"]
