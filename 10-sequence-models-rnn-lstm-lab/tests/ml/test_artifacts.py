import uuid
from pathlib import Path

import numpy as np
import pytest
import torch

from sequence_models.artifacts import BundleError, load_bundle, save_bundle
from sequence_models.contracts import ModelConfig
from sequence_models.inference import SequencePredictor
from sequence_models.models import build_model


def runtime_path() -> Path:
    path = Path("tests/runtime") / f"artifact-{uuid.uuid4().hex}"
    path.mkdir(parents=True)
    return path


def test_bundle_roundtrip_preserves_prediction() -> None:
    torch.manual_seed(7)
    config = ModelConfig(model_type="gru", hidden_size=8, dropout=0.0)
    model = build_model(config).eval()
    values = np.ones((128, 9), dtype=np.float32)
    before = SequencePredictor(model).predict(values)
    path = save_bundle(
        runtime_path() / "bundle",
        model,
        config,
        {"macro_f1": 0.0},
        {"mean": [0.0] * 9, "std": [1.0] * 9},
        {"model_id": "gru", "version": "test"},
    )
    loaded = load_bundle(path)
    after = SequencePredictor(loaded.model).predict(values)
    assert before.predicted_index == after.predicted_index
    assert np.allclose(before.probabilities, after.probabilities)
    assert not loaded.model.training


def test_bundle_rejects_tampered_state() -> None:
    config = ModelConfig(model_type="rnn", hidden_size=4, dropout=0.0)
    path = save_bundle(
        runtime_path() / "bundle",
        build_model(config),
        config,
        {},
        {},
        {"model_id": "rnn", "version": "test"},
    )
    with (path / "model_state.pt").open("ab") as handle:
        handle.write(b"tampered")
    with pytest.raises(BundleError, match="hash"):
        load_bundle(path)
