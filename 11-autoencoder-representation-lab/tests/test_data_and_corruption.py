from __future__ import annotations

import numpy as np
import pytest
import torch

from autoencoder_lab.corruption import corrupt
from autoencoder_lab.data import build_fixture_records, stack_records


def test_fixture_is_deterministic() -> None:
    first = build_fixture_records(6)
    second = build_fixture_records(6)
    assert first[0].sample_id == second[0].sample_id
    assert np.array_equal(first[0].image, second[0].image)


def test_fixture_split_contract() -> None:
    records = build_fixture_records(15)
    train, train_labels, train_ids = stack_records(records, "train")
    validation, _, _ = stack_records(records, "validation")
    test, _, _ = stack_records(records, "test")
    assert train.shape == (100, 1, 28, 28)
    assert validation.shape == (30, 1, 28, 28)
    assert test.shape == (20, 1, 28, 28)
    assert train.dtype == np.float32
    assert len(set(train_ids)) == len(train_ids)
    assert set(train_labels) == set(range(10))


@pytest.mark.parametrize("corruption_type", ["gaussian", "masking"])
def test_corruption_is_bounded_and_deterministic(corruption_type: str) -> None:
    images = torch.full((2, 1, 28, 28), 0.5)
    first = corrupt(images, corruption_type, 0.2, seed=7)  # type: ignore[arg-type]
    second = corrupt(images, corruption_type, 0.2, seed=7)  # type: ignore[arg-type]
    assert torch.equal(first, second)
    assert float(first.min()) >= 0
    assert float(first.max()) <= 1
    assert not torch.equal(first, images)


def test_corruption_rejects_unapproved_level() -> None:
    with pytest.raises(ValueError, match="level"):
        corrupt(torch.zeros(1, 1, 28, 28), "gaussian", 0.15)


def test_fixture_rejects_too_few_samples() -> None:
    with pytest.raises(ValueError, match="between 6 and 30"):
        build_fixture_records(5)
