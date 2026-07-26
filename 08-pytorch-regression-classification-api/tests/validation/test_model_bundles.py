from __future__ import annotations

import json

import pytest

from pytorch_tabular.artifacts import build_bundle, validate_bundle
from pytorch_tabular.contracts import ExperimentConfig
from pytorch_tabular.experiments import run_experiment
from pytorch_tabular.inference import ModelPredictor


def test_bundle_reload_preserves_classification_predictions(runtime_dir) -> None:
    result = run_experiment(
        ExperimentConfig(task="classification", epochs=50, patience=10),
        runtime_dir,
    )
    bundle = build_bundle(result, runtime_dir / "bundle")
    predictor = ModelPredictor(bundle)
    example = json.loads((bundle / "metadata.json").read_text())["examples"][0]
    first = predictor.predict([example])
    second = ModelPredictor(bundle).predict([example])
    assert first == second


def test_incomplete_bundle_is_rejected(runtime_dir) -> None:
    (runtime_dir / "metadata.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="Incomplete"):
        validate_bundle(runtime_dir)


def test_batch_limit_is_enforced(runtime_dir) -> None:
    result = run_experiment(
        ExperimentConfig(task="classification", epochs=30, patience=8),
        runtime_dir,
    )
    bundle = build_bundle(result, runtime_dir / "bundle")
    predictor = ModelPredictor(bundle)
    example = predictor.metadata["examples"][0]
    with pytest.raises(ValueError, match="100"):
        predictor.predict([example] * 101)
