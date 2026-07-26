from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import json
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

from pytorch_tabular.artifacts import validate_bundle
from pytorch_tabular.models import ClassificationMLP, RegressionMLP


class ModelPredictor:
    def __init__(self, bundle: Path) -> None:
        self.bundle = bundle
        self.metadata: dict[str, Any] = validate_bundle(bundle)
        architecture = cast(dict[str, Any], self.metadata["architecture"])
        if self.metadata["task"] == "regression":
            self.model: torch.nn.Module = RegressionMLP(
                input_features=architecture["input_features"],
                hidden_units=tuple(architecture["hidden_units"]),
                dropout=architecture["dropout"],
            )
        else:
            self.model = ClassificationMLP(
                input_features=architecture["input_features"],
                class_count=architecture["output_units"],
                hidden_units=tuple(architecture["hidden_units"]),
                dropout=architecture["dropout"],
            )
        state_dict = torch.load(
            bundle / "model_state.pt", map_location="cpu", weights_only=True
        )
        self.model.load_state_dict(state_dict)
        self.model.eval()
        preprocessing = json.loads(
            (bundle / "preprocessing.json").read_text(encoding="utf-8")
        )
        self.scaler = StandardScaler()
        self.scaler.mean_ = np.asarray(preprocessing["mean"], dtype=np.float64)
        self.scaler.scale_ = np.asarray(preprocessing["scale"], dtype=np.float64)
        self.scaler.var_ = np.asarray(preprocessing["variance"], dtype=np.float64)
        self.scaler.n_features_in_ = int(preprocessing["feature_count"])
        self.scaler.n_samples_seen_ = int(preprocessing["samples_seen"])

    @property
    def task(self) -> str:
        return str(self.metadata["task"])

    def _matrix(self, rows: list[dict[str, float]]) -> np.ndarray:
        expected = cast(list[str], self.metadata["feature_names"])
        matrix: list[list[float]] = []
        for row in rows:
            if set(row) != set(expected):
                raise ValueError("Feature names do not match the active model schema.")
            values = [float(row[name]) for name in expected]
            if not np.isfinite(values).all():
                raise ValueError("All features must contain finite numeric values.")
            matrix.append(values)
        return self.scaler.transform(np.asarray(matrix, dtype=np.float32)).astype(np.float32)

    def predict(self, rows: list[dict[str, float]]) -> list[dict[str, object]]:
        if not rows:
            raise ValueError("At least one observation is required.")
        if len(rows) > 100:
            raise ValueError("Batch inference accepts at most 100 observations.")
        features = torch.as_tensor(self._matrix(rows), dtype=torch.float32)
        with torch.inference_mode():
            output = self.model(features)
            if self.task == "regression":
                return [
                    {
                        "value": float(value),
                        "unit": self.metadata["target_unit"],
                    }
                    for value in output.tolist()
                ]
            probabilities = torch.softmax(output, dim=1).tolist()
            class_names = cast(list[str], self.metadata["class_names"])
            return [
                {
                    "class_name": class_names[int(np.argmax(row))],
                    "probabilities": {
                        name: float(probability)
                        for name, probability in zip(class_names, row)
                    },
                }
                for row in probabilities
            ]
