from __future__ import annotations

import json

import numpy as np
import pytest
from pydantic import ValidationError

from cnn_foundations.contracts.config import DatasetConfig, ModelConfig
from cnn_foundations.evaluation.error_analysis import build_error_records
from cnn_foundations.evaluation.metrics import evaluate_predictions


def test_dataset_contract_has_stable_classes() -> None:
    assert len(DatasetConfig().class_names) == 10


def test_invalid_cnn_contract_is_rejected() -> None:
    with pytest.raises(ValidationError):
        ModelConfig(kind="cnn", input_channels=3, conv_channels=(8, 16))


def test_metrics_include_all_ten_classes() -> None:
    report = evaluate_predictions([0, 1, 1], [0, 1, 0])
    assert report.accuracy == pytest.approx(2 / 3)
    assert len(report.per_class) == 10
    assert len(report.confusion_matrix) == 10


def test_metrics_report_is_json_serializable() -> None:
    report = evaluate_predictions([0, 1, 1], [0, 1, 0])

    encoded = json.dumps(report.as_dict())

    assert '"index": 0' in encoded


def test_error_analysis_contains_only_mistakes() -> None:
    probabilities = np.zeros((2, 10))
    probabilities[0, 0] = 1
    probabilities[1, 2] = 1
    rows = build_error_records([0, 1], probabilities, ["a", "b"])
    assert [row["sample_id"] for row in rows] == ["b"]
