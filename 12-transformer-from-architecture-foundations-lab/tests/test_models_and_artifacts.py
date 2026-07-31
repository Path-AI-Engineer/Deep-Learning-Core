from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

import pytest
import torch

from transformer_lab.artifacts import JSON_FILES, load_bundle, write_bundle
from transformer_lab.contracts import ModelConfig
from transformer_lab.models import GRUEncoderDecoder, SequenceTransformer
from transformer_lab.training import sequence_loss


def config() -> ModelConfig:
    return ModelConfig(
        vocabulary_size=39,
        d_model=16,
        num_heads=4,
        encoder_layers=1,
        decoder_layers=1,
        d_ff=32,
        max_length=16,
    )


def test_transformer_forward_and_trace_shapes() -> None:
    model = SequenceTransformer(config())
    source = torch.tensor([[4, 7, 8, 2], [5, 9, 10, 2]])
    target = torch.tensor([[1, 7, 8], [1, 10, 9]])
    logits, trace = model(source, target, trace=True)
    assert logits.shape == (2, 3, 39)
    assert trace["memory_shape"] == [2, 4, 16]
    assert len(trace["encoder_self"]) == 1
    assert trace["encoder_self"][0].shape == (2, 4, 4, 4)
    assert trace["decoder_self"][0].shape == (2, 4, 3, 3)
    assert trace["cross"][0].shape == (2, 4, 3, 4)


def test_gru_baseline_uses_same_sequence_contract() -> None:
    model = GRUEncoderDecoder(config())
    logits = model(
        torch.tensor([[4, 7, 8, 2]]),
        torch.tensor([[1, 7, 8]]),
    )
    assert logits.shape == (1, 3, 39)


def test_sequence_loss_ignores_padding() -> None:
    logits = torch.randn(1, 3, 39)
    target = torch.tensor([[7, 2, 0]])
    assert torch.isfinite(sequence_loss(logits, target))
    with pytest.raises(ValueError, match="share batch"):
        sequence_loss(logits[:, :2], target)


def test_bundle_round_trip_and_hash_tamper_detection() -> None:
    destination = Path(".test-artifacts") / f"bundle-{uuid4().hex}"
    documents = {filename: {} for filename in JSON_FILES}
    documents["model_config.json"] = config().as_dict()
    write_bundle(
        destination,
        SequenceTransformer(config()),
        documents=documents,
        metadata={
            "model_version": "test",
            "seed": 12,
            "evidence_status": "test_fixture",
            "limitations": [],
        },
    )
    loaded = load_bundle(destination)
    assert loaded.manifest["model_version"] == "test"
    (destination / "metrics.json").write_text(json.dumps({"tampered": True}))
    with pytest.raises(ValueError, match="hash mismatch"):
        load_bundle(destination)
