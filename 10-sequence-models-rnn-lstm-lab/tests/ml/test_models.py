import pytest
import torch

from sequence_models.contracts import ModelConfig
from sequence_models.models import build_model, count_parameters


@pytest.mark.parametrize("model_type", ["statistics-mlp", "rnn", "lstm", "gru"])
def test_model_forward_returns_six_logits(model_type: str) -> None:
    model = build_model(ModelConfig(model_type=model_type))
    logits = model(torch.randn(3, 128, 9))
    assert logits.shape == (3, 6)
    assert count_parameters(model) > 0


@pytest.mark.parametrize("model_type", ["rnn", "lstm", "gru"])
def test_recurrent_model_supports_batch_size_one(model_type: str) -> None:
    model = build_model(ModelConfig(model_type=model_type))
    assert model(torch.randn(1, 128, 9)).shape == (1, 6)


def test_recurrent_model_uses_packed_valid_lengths() -> None:
    model = build_model(ModelConfig(model_type="lstm")).eval()
    padded = torch.randn(2, 8, 9)
    lengths = torch.tensor([8, 4])
    logits = model(padded, lengths)
    changed_padding = padded.clone()
    changed_padding[1, 4:] = 999.0
    second_logits = model(changed_padding, lengths)
    assert torch.allclose(logits[1], second_logits[1], atol=1e-6)


def test_invalid_input_size_is_rejected() -> None:
    with pytest.raises(ValueError, match="nine UCI HAR"):
        ModelConfig(model_type="rnn", input_size=8).validate()
