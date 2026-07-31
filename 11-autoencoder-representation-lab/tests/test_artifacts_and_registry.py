from __future__ import annotations

from pathlib import Path

import pytest

from app.core import LabRegistry
from autoencoder_lab.artifacts import BundleError, load_bundle

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_committed_bundle_loads_and_is_in_eval_mode() -> None:
    bundle = load_bundle(PROJECT_ROOT / "artifacts/models/conv-ae/v1.0.0")
    assert bundle.model_id == "conv-ae"
    assert bundle.version == "v1.0.0"
    assert bundle.model.training is False


def test_bundle_rejects_missing_state() -> None:
    with pytest.raises(BundleError, match="state"):
        load_bundle(PROJECT_ROOT / "tests/fixtures/incomplete_bundle")


def test_registry_reports_explicit_fixture_mode() -> None:
    registry = LabRegistry()
    assert registry.data_mode == "educational_fixture"
    assert registry.active_model == "conv-ae"
    assert set(registry.bundles) == {
        "dense-ae",
        "conv-ae",
        "denoising-ae",
        "latent-2d",
    }


def test_registry_reconstructs_baseline_and_neural_model() -> None:
    registry = LabRegistry()
    baseline = registry.reconstruct(registry.sample_ids[0], "pca")
    neural = registry.reconstruct(registry.sample_ids[0], "conv-ae")
    assert baseline["metrics"]["mse"] >= 0  # type: ignore[index]
    assert len(neural["latent"]) == 16  # type: ignore[arg-type]
    assert str(neural["reconstruction"]).startswith("data:image/png;base64,")


def test_registry_rejects_unknown_model() -> None:
    with pytest.raises(ValueError, match="unknown model_id"):
        LabRegistry().require_model("imaginary-ae")


def test_latent_decode_enforces_observed_bounds() -> None:
    registry = LabRegistry()
    with pytest.raises(ValueError, match="outside observed"):
        registry.decode(1_000.0, 1_000.0)
