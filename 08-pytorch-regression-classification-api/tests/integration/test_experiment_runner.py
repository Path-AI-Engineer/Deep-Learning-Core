from __future__ import annotations

import json

from pytorch_tabular.contracts import ExperimentConfig
from pytorch_tabular.experiments import run_experiment


def test_runner_creates_non_overwriting_evidence(runtime_dir) -> None:
    config = ExperimentConfig(
        task="classification",
        epochs=30,
        patience=8,
        batch_size=24,
    )
    first = run_experiment(config, runtime_dir)
    second = run_experiment(config, runtime_dir)
    assert first.run_id != second.run_id
    payload = json.loads((first.run_directory / "run.json").read_text(encoding="utf-8"))
    assert payload["config"]["seed"] == 42
    assert payload["environment"]["device"] == "cpu"
