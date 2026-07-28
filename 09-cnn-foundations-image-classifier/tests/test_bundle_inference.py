from __future__ import annotations

import io
from pathlib import Path

import pytest
from PIL import Image

from cnn_foundations.artifacts.bundle import (
    JSON_FILES,
    BundleContents,
    load_bundle,
    write_bundle,
)
from cnn_foundations.explanations.activations import ActivationInspector
from cnn_foundations.inference.predictor import Predictor
from cnn_foundations.inference.preprocessing import process_image_bytes
from cnn_foundations.models.cnn import FashionCNN


def documents() -> dict[str, object]:
    return {
        "model_config.json": {
            "kind": "cnn",
            "num_classes": 10,
            "hidden_features": 16,
            "dropout": 0,
            "input_channels": 1,
            "input_features": None,
            "conv_channels": [4, 8],
            "batch_norm": False,
        },
        "preprocessing.json": {"mean": 0.286, "std": 0.353},
        "class_mapping.json": {str(index): str(index) for index in range(10)},
        "metrics.json": {"accuracy": 0.1},
        "per_class_metrics.json": {"classes": []},
        "confusion_matrix.json": {"matrix": []},
        "training_history.json": {"epochs": []},
        "comparison_with_mlp.json": {},
        "error_analysis.json": {"errors": []},
        "split_manifest.json": {},
    }


def test_predictor_uses_bundle_contract() -> None:
    model = FashionCNN(
        channels=(4, 8), hidden_features=16, dropout=0, batch_norm=False
    )
    model.eval()
    bundle = BundleContents(
        model=model,
        metadata={"model_version": "v-test"},
        files=documents(),
    )
    prediction = Predictor(bundle).predict(
        process_image_bytes(image_bytes(), mime_type="image/png", mean=0.286, std=0.353).tensor
    )
    assert len(prediction.probabilities) == 10
    assert sum(row["probability"] for row in prediction.probabilities) == pytest.approx(1.0)
    assert prediction.model_version == "v-test"
    assert len(JSON_FILES) == 10


def test_bundle_rejects_incomplete_contract() -> None:
    with pytest.raises(ValueError, match="documents are missing"):
        write_bundle(
            Path("tests/fixtures/not-created"),
            FashionCNN(
                channels=(4, 8),
                hidden_features=16,
                dropout=0,
                batch_norm=False,
            ),
            metadata={"model_version": "v-test"},
            documents={},
        )


def test_loader_rejects_missing_bundle() -> None:
    with pytest.raises(ValueError, match="incomplete CNN bundle"):
        load_bundle(Path("tests/fixtures/missing-bundle"))


def image_bytes() -> bytes:
    buffer = io.BytesIO()
    Image.new("L", (28, 28), color=128).save(buffer, format="PNG")
    return buffer.getvalue()


def test_preprocessing_contract() -> None:
    processed = process_image_bytes(
        image_bytes(), mime_type="image/png", mean=0.286, std=0.353
    )
    assert tuple(processed.tensor.shape) == (1, 1, 28, 28)
    assert processed.preview_data_url.startswith("data:image/png;base64,")


def test_preprocessing_rejects_wrong_media_type() -> None:
    with pytest.raises(TypeError):
        process_image_bytes(b"data", mime_type="text/plain", mean=0.286, std=0.353)


def test_activation_inspector_removes_hook() -> None:
    model = FashionCNN(channels=(4, 8), hidden_features=16, dropout=0, batch_norm=False)
    inspector = ActivationInspector(model)
    result = inspector.capture(
        process_image_bytes(
            image_bytes(), mime_type="image/png", mean=0.286, std=0.353
        ).tensor,
        layer_id="conv1",
        limit=2,
    )
    assert len(result["feature_maps"]) == 2
    assert not model.conv1._forward_hooks
