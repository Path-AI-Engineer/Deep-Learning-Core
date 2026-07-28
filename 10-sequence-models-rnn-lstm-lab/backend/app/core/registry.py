from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch

from sequence_models.artifacts import ModelBundle, load_bundle
from sequence_models.cells import cell_trace
from sequence_models.contracts import CLASS_DESCRIPTIONS, CLASS_MAPPING
from sequence_models.data import SequenceRecord, build_demo_records
from sequence_models.experiments import gradient_flow_experiment
from sequence_models.inference import SequencePredictor
from sequence_models.models import count_parameters

PROJECT_ROOT = Path(__file__).resolve().parents[3]


class LabRegistry:
    def __init__(self) -> None:
        self.data_mode = os.getenv("SEQUENCE_DATA_MODE", "fixture")
        self.records = build_demo_records(4)
        self.samples = {record.sample_id: record for record in self.records}
        self.bundles: dict[str, ModelBundle] = {}
        bundle_root = Path(os.getenv("SEQUENCE_BUNDLE_ROOT", PROJECT_ROOT / "artifacts/models"))
        if not bundle_root.is_absolute():
            bundle_root = PROJECT_ROOT / bundle_root
        for model_id in ("rnn", "lstm", "gru"):
            path = bundle_root / model_id / "v1.0.0"
            if path.is_dir():
                self.bundles[model_id] = load_bundle(path)
        comparison_path = PROJECT_ROOT / "artifacts/comparisons/v1.0.0/model_comparison.json"
        self.comparison = (
            json.loads(comparison_path.read_text(encoding="utf-8"))
            if comparison_path.is_file()
            else {
                "approved_model": None,
                "models": [],
                "warning": "No comparison artifact is available.",
            }
        )
        approved = self.comparison.get("approved_model")
        self.active_model = (
            str(approved)
            if approved in self.bundles
            else (next(iter(self.bundles)) if self.bundles else None)
        )

    def require_sample(self, sample_id: str) -> SequenceRecord:
        try:
            return self.samples[sample_id]
        except KeyError as exc:
            raise KeyError(f"unknown sample_id: {sample_id}") from exc

    def require_model(self, model_id: str) -> tuple[str, ModelBundle]:
        resolved = self.active_model if model_id == "active" else model_id
        if resolved is None or resolved not in self.bundles:
            raise KeyError(f"model is not available: {model_id}")
        return resolved, self.bundles[resolved]

    def predict(self, sample_id: str, model_id: str) -> dict[str, Any]:
        sample = self.require_sample(sample_id)
        resolved, bundle = self.require_model(model_id)
        preprocessing = bundle.preprocessing
        mean = np.asarray(preprocessing.get("mean", [0.0] * 9), dtype=np.float32)
        std = np.asarray(preprocessing.get("std", [1.0] * 9), dtype=np.float32)
        result = SequencePredictor(bundle.model, mean, std).predict(sample.values)
        top_indices = np.argsort(result.probabilities)[::-1][:3]
        return {
            "prediction_id": result.prediction_id,
            "sample_id": sample_id,
            "true_class": sample.activity,
            "predicted_class": result.predicted_class,
            "probabilities": [
                {"class_name": CLASS_MAPPING[index], "probability": result.probabilities[index]}
                for index in CLASS_MAPPING
            ],
            "top_k": [
                {
                    "class_name": CLASS_MAPPING[int(index)],
                    "probability": result.probabilities[int(index)],
                }
                for index in top_indices
            ],
            "confidence": result.confidence,
            "model_type": resolved,
            "model_version": bundle.version,
            "latency_ms": round(result.latency_ms, 4),
            "warnings": [
                "Probability is model confidence, not certainty.",
                "Fixture mode validates the system and is not a UCI HAR benchmark.",
            ]
            if self.data_mode == "fixture"
            else ["Probability is model confidence, not certainty."],
        }

    def sample_trace(
        self,
        sample_id: str,
        model_id: str,
        units: list[int],
        start: int,
        end: int,
    ) -> dict[str, Any]:
        sample = self.require_sample(sample_id)
        resolved, bundle = self.require_model(model_id)
        model = bundle.model
        recurrent: Any = cast(Any, model).recurrent
        values = torch.from_numpy(sample.values).unsqueeze(0)
        hidden = torch.zeros(1, 24)
        cell = torch.zeros(1, 24)
        hidden_norms: list[float] = []
        cell_norms: list[float] = []
        selected: list[list[float]] = []
        with torch.inference_mode():
            for timestep in range(values.shape[1]):
                current = values[:, timestep, :]
                if resolved == "rnn":
                    hidden = torch.tanh(
                        current @ recurrent.weight_ih_l0.T
                        + recurrent.bias_ih_l0
                        + hidden @ recurrent.weight_hh_l0.T
                        + recurrent.bias_hh_l0
                    )
                elif resolved == "lstm":
                    gates = (
                        current @ recurrent.weight_ih_l0.T
                        + recurrent.bias_ih_l0
                        + hidden @ recurrent.weight_hh_l0.T
                        + recurrent.bias_hh_l0
                    )
                    input_gate, forget_gate, candidate, output_gate = gates.chunk(4, dim=1)
                    cell = (
                        torch.sigmoid(forget_gate) * cell
                        + torch.sigmoid(input_gate) * torch.tanh(candidate)
                    )
                    hidden = torch.sigmoid(output_gate) * torch.tanh(cell)
                    cell_norms.append(float(cell.norm().item()))
                else:
                    input_gates = current @ recurrent.weight_ih_l0.T + recurrent.bias_ih_l0
                    hidden_gates = hidden @ recurrent.weight_hh_l0.T + recurrent.bias_hh_l0
                    input_r, input_z, input_n = input_gates.chunk(3, dim=1)
                    hidden_r, hidden_z, hidden_n = hidden_gates.chunk(3, dim=1)
                    reset = torch.sigmoid(input_r + hidden_r)
                    update = torch.sigmoid(input_z + hidden_z)
                    new = torch.tanh(input_n + reset * hidden_n)
                    hidden = (1 - update) * new + update * hidden
                hidden_norms.append(float(hidden.norm().item()))
                selected.append([float(hidden[0, unit].item()) for unit in units])
        bounded_end = min(end, 127)
        return {
            "sample_id": sample_id,
            "model_id": resolved,
            "timesteps": list(range(start, bounded_end + 1)),
            "hidden_norms": hidden_norms[start : bounded_end + 1],
            "cell_norms": cell_norms[start : bounded_end + 1] if cell_norms else None,
            "selected_units": units,
            "selected_values": selected[start : bounded_end + 1],
            "prediction": self.predict(sample_id, resolved),
            "warning": "Internal states are observations, not causal explanations.",
        }

    def model_rows(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        metrics_by_id = {
            item["model_id"]: item for item in self.comparison.get("models", [])
        }
        for model_id, bundle in self.bundles.items():
            metrics = metrics_by_id.get(model_id, bundle.metrics)
            rows.append(
                {
                    "model_id": model_id,
                    "type": model_id.upper(),
                    "version": bundle.version,
                    "hidden_size": bundle.manifest.get("hidden_size", 24),
                    "parameters": metrics.get("parameters", count_parameters(bundle.model)),
                    "metrics": {
                        "accuracy": metrics.get("accuracy"),
                        "macro_f1": metrics.get("macro_f1"),
                        "validation_macro_f1": metrics.get("validation_macro_f1"),
                    },
                    "latency_ms": metrics.get("latency_ms"),
                    "available": True,
                    "active": model_id == self.active_model,
                }
            )
        return rows

    @staticmethod
    def classes() -> list[dict[str, object]]:
        return [
            {
                "index": index,
                "label": label,
                "description": CLASS_DESCRIPTIONS[index],
            }
            for index, label in CLASS_MAPPING.items()
        ]

    @staticmethod
    def cell_trace(cell_type: str) -> dict[str, object]:
        return cell_trace(cell_type).to_dict()  # type: ignore[arg-type]

    @staticmethod
    def gradient_flow() -> dict[str, object]:
        return gradient_flow_experiment()


@lru_cache(maxsize=1)
def get_registry() -> LabRegistry:
    return LabRegistry()
