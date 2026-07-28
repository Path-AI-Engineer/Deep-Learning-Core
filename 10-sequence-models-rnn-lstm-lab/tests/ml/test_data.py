import uuid
from pathlib import Path

import numpy as np
import pytest

from sequence_models.contracts import CHANNELS
from sequence_models.data import build_demo_records, load_uci_har, prepare_grouped_splits


def test_demo_records_follow_sequence_contract() -> None:
    records = build_demo_records(2)
    assert len(records) == 12
    assert all(record.values.shape == (128, 9) for record in records)
    assert all(record.values.dtype == np.float32 for record in records)
    assert all(np.isfinite(record.values).all() for record in records)


def _write_split(root: Path, split: str, subjects: list[int]) -> None:
    split_root = root / split
    signal_root = split_root / "Inertial Signals"
    signal_root.mkdir(parents=True)
    rows = len(subjects)
    for index, channel in enumerate(CHANNELS):
        matrix = np.full((rows, 128), index + 1, dtype=np.float32)
        np.savetxt(signal_root / f"{channel}_{split}.txt", matrix)
    np.savetxt(split_root / f"y_{split}.txt", np.arange(rows) % 6 + 1, fmt="%d")
    np.savetxt(split_root / f"subject_{split}.txt", subjects, fmt="%d")


def runtime_path() -> Path:
    path = Path("tests/runtime") / f"data-{uuid.uuid4().hex}"
    path.mkdir(parents=True)
    return path


def test_uci_parser_and_group_split_are_leakage_safe() -> None:
    root = runtime_path()
    dataset = root / "UCI HAR Dataset"
    _write_split(dataset, "train", [1, 2, 3, 4, 5, 6])
    _write_split(dataset, "test", [21, 22, 23])
    official_train, official_test = load_uci_har(root)
    prepared = prepare_grouped_splits(official_train, official_test, (1, 3, 5))
    assert prepared.train.values.shape[1:] == (128, 9)
    assert set(prepared.train.subjects).isdisjoint(prepared.validation.subjects)
    assert set(prepared.test.subjects).isdisjoint(official_train.subjects)
    assert prepared.train.values.dtype == np.float32


def test_missing_uci_extraction_has_actionable_error() -> None:
    with pytest.raises(FileNotFoundError, match="UCI HAR extraction"):
        load_uci_har(runtime_path())
