from __future__ import annotations

import pytest
import torch

from autoencoder_lab.contracts import CLASS_MAPPING, ModelConfig, TrainingConfig
from autoencoder_lab.models import assert_autoencoder_contract, build_model, count_parameters


def test_class_mapping_is_complete() -> None:
    assert list(CLASS_MAPPING) == list(range(10))
    assert CLASS_MAPPING[9] == "Ankle boot"


@pytest.mark.parametrize(
    ("model_type", "latent_dim"),
    [
        ("dense-ae", 16),
        ("conv-ae", 16),
        ("denoising-ae", 16),
        ("latent-2d", 2),
    ],
)
def test_model_contract(model_type: str, latent_dim: int) -> None:
    config = ModelConfig(model_type=model_type, latent_dim=latent_dim)  # type: ignore[arg-type]
    model = build_model(config)
    assert_autoencoder_contract(model, batch_size=2)
    assert count_parameters(model) > 1_000


def test_latent_two_requires_two_dimensions() -> None:
    with pytest.raises(ValueError, match="requires latent_dim=2"):
        ModelConfig(model_type="latent-2d", latent_dim=16).validate()


def test_model_rejects_unknown_latent_dimension() -> None:
    with pytest.raises(ValueError, match="latent_dim"):
        ModelConfig(model_type="dense-ae", latent_dim=3).validate()


def test_training_contract_rejects_non_positive_values() -> None:
    with pytest.raises(ValueError, match="positive"):
        TrainingConfig(epochs=0).validate()


def test_encoder_decoder_preserves_batch() -> None:
    model = build_model(ModelConfig(model_type="conv-ae", latent_dim=8))
    inputs = torch.rand(4, 1, 28, 28)
    latent = model.encode(inputs)
    assert latent.shape == (4, 8)
    assert model.decode(latent).shape == inputs.shape
