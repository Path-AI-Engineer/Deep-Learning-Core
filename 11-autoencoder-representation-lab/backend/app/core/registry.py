from __future__ import annotations

import json
import math
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from torch import Tensor

from autoencoder_lab.artifacts import AutoencoderBundle, load_bundle
from autoencoder_lab.contracts import CLASS_MAPPING
from autoencoder_lab.corruption import corrupt
from autoencoder_lab.evaluation import per_sample_mse, reconstruction_metrics
from autoencoder_lab.inference import image_data_url, reconstruct
from autoencoder_lab.protocols import EncoderDecoder
from autoencoder_lab.representations import interpolate

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path.name} must contain an object")
    return payload


class LabRegistry:
    def __init__(self) -> None:
        version = os.getenv("LATENT_LAB_ARTIFACT_VERSION", "v1.0.0")
        artifact_root = PROJECT_ROOT / "artifacts"
        self.version = version
        self.comparison = _json(
            artifact_root / "comparisons" / version / "model_comparison.json"
        )
        self.active_model = str(
            os.getenv("LATENT_LAB_ACTIVE_MODEL", self.comparison["active_model"])
        )
        self.bundles: dict[str, AutoencoderBundle] = {}
        for model_id in ("dense-ae", "conv-ae", "denoising-ae", "latent-2d"):
            self.bundles[model_id] = load_bundle(
                artifact_root / "models" / model_id / version
            )
        mean_archive = np.load(
            artifact_root / "models" / "mean-image" / version / "baseline.npz"
        )
        self.mean_image = mean_archive["mean_image"].astype(np.float32)
        pca_archive = np.load(artifact_root / "models" / "pca" / version / "pca.npz")
        self.pca_mean = pca_archive["mean"].astype(np.float32)
        self.pca_components = pca_archive["components"].astype(np.float32)
        fixture = _json(PROJECT_ROOT / "data" / "samples" / "fixture_tensors.json")
        self.sample_ids = [str(value) for value in fixture["sample_ids"]]
        self.labels = np.asarray(fixture["labels"], dtype=np.int64)
        self.images = torch.tensor(fixture["images"], dtype=torch.float32)
        self.gallery = _json(
            PROJECT_ROOT / "data" / "samples" / "fixture_gallery.json"
        )["items"]
        self.latent_index = _json(
            artifact_root / "models" / "latent-2d" / version / "sample_index.json"
        )["items"]
        self.latent_bounds = _json(
            artifact_root / "models" / "latent-2d" / version / "latent_bounds.json"
        )
        self._sample_lookup = {
            sample_id: index for index, sample_id in enumerate(self.sample_ids)
        }

    @property
    def data_mode(self) -> str:
        return str(self.comparison["data_mode"])

    def require_model(self, model_id: str) -> None:
        if model_id not in {"mean-image", "pca", *self.bundles}:
            raise ValueError(f"unknown model_id: {model_id}")

    def sample(self, sample_id: str) -> tuple[Tensor, int]:
        if sample_id not in self._sample_lookup:
            raise ValueError(f"unknown sample_id: {sample_id}")
        index = self._sample_lookup[sample_id]
        return self.images[index : index + 1], int(self.labels[index])

    def classes(self) -> list[dict[str, int | str]]:
        return [
            {"class_id": class_id, "class_name": class_name}
            for class_id, class_name in CLASS_MAPPING.items()
        ]

    def model_rows(self) -> list[dict[str, object]]:
        rows = []
        for row in self.comparison["models"]:
            model_id = str(row["model_id"])
            rows.append(
                {
                    **row,
                    "version": self.version,
                    "active": model_id == self.active_model,
                    "capabilities": {
                        "reconstruct": True,
                        "denoise": model_id in ("conv-ae", "denoising-ae"),
                        "encode": model_id not in ("mean-image",),
                        "decode_coordinates": model_id == "latent-2d",
                    },
                }
            )
        return rows

    def _baseline_reconstruct(self, model_id: str, image: Tensor) -> Tensor:
        values = image.detach().cpu().numpy()
        if model_id == "mean-image":
            return torch.from_numpy(np.repeat(self.mean_image[None, ...], len(values), axis=0))
        flat = values.reshape(len(values), -1)
        latent = (flat - self.pca_mean) @ self.pca_components.T
        reconstructed = latent @ self.pca_components + self.pca_mean
        return torch.from_numpy(reconstructed.reshape(-1, 1, 28, 28).astype(np.float32))

    def reconstruct(self, sample_id: str, model_id: str) -> dict[str, object]:
        self.require_model(model_id)
        image, label = self.sample(sample_id)
        if model_id in ("mean-image", "pca"):
            prediction = self._baseline_reconstruct(model_id, image)
            result: dict[str, object] = {
                "original": image_data_url(image[0]),
                "reconstruction": image_data_url(prediction[0]),
                "absolute_error": image_data_url(torch.abs(prediction - image)[0]),
                "latent": [],
                "metrics": reconstruction_metrics(image, prediction).to_dict(),
            }
        else:
            result = reconstruct(self.bundles[model_id].model, image)
        return {
            **result,
            "sample_id": sample_id,
            "label": label,
            "class_name": CLASS_MAPPING[label],
            "model_id": model_id,
            "model_version": self.version,
            "data_mode": self.data_mode,
            "latency_ms": None,
            "warning": "Fixture evidence; not FashionMNIST benchmark performance.",
        }

    def denoise(
        self,
        sample_id: str,
        corruption_type: str,
        level: float,
        seed: int,
        model_ids: list[str],
    ) -> dict[str, object]:
        clean, label = self.sample(sample_id)
        corrupted = corrupt(clean, corruption_type, level, seed)  # type: ignore[arg-type]
        rows = []
        for model_id in model_ids:
            bundle = self.bundles[model_id]
            bundle.model.eval()
            with torch.inference_mode():
                prediction = bundle.model(corrupted)
            rows.append(
                {
                    "model_id": model_id,
                    "image": image_data_url(prediction[0]),
                    "metrics": reconstruction_metrics(clean, prediction).to_dict(),
                }
            )
        return {
            "sample_id": sample_id,
            "label": label,
            "clean": image_data_url(clean[0]),
            "corrupted": image_data_url(corrupted[0]),
            "corruption": {
                "type": corruption_type,
                "level": level,
                "seed": seed,
                "target": "clean",
            },
            "reconstructions": rows,
            "warning": "Robustness is limited to the evaluated corruption family and levels.",
        }

    def latent_points(
        self,
        class_ids: set[int] | None,
        limit: int,
    ) -> dict[str, object]:
        if not 1 <= limit <= 300:
            raise ValueError("limit must be between 1 and 300")
        items = [
            item
            for item in self.latent_index
            if class_ids is None or int(item["label"]) in class_ids
        ][:limit]
        return {
            "items": items,
            "count": len(items),
            "bounds": self.latent_bounds,
            "model_id": "latent-2d",
            "warning": "Class color is evaluation metadata, not a training signal.",
        }

    def latent_sample(self, sample_id: str) -> dict[str, object]:
        for item in self.latent_index:
            if item["sample_id"] == sample_id:
                return {**item, "bounds": self.latent_bounds}
        raise ValueError(f"unknown sample_id: {sample_id}")

    def decode(self, x: float, y: float) -> dict[str, object]:
        if not math.isfinite(x) or not math.isfinite(y):
            raise ValueError("latent coordinates must be finite")
        x_bounds, y_bounds = self.latent_bounds["x"], self.latent_bounds["y"]
        if not x_bounds[0] <= x <= x_bounds[1] or not y_bounds[0] <= y <= y_bounds[1]:
            raise ValueError("latent coordinate is outside observed training bounds")
        model = self.bundles["latent-2d"].model
        latent = torch.tensor([[x, y]], dtype=torch.float32)
        with torch.inference_mode():
            decoded = cast(EncoderDecoder, model).decode(latent)
        points = np.asarray([[item["x"], item["y"]] for item in self.latent_index])
        distance = float(np.linalg.norm(points - np.asarray([x, y]), axis=1).min())
        return {
            "model_id": "latent-2d",
            "model_version": self.version,
            "coordinate": {"x": x, "y": y},
            "image": image_data_url(decoded[0]),
            "nearest_observed_distance": round(distance, 6),
            "support_warning": (
                "Low-support coordinate; decoder output is exploratory."
                if distance > 1.0
                else "Coordinate is near an observed fixture sample."
            ),
        }

    def interpolation(
        self,
        model_id: str,
        sample_id_a: str,
        sample_id_b: str,
        steps: int,
    ) -> dict[str, object]:
        if model_id not in self.bundles:
            raise ValueError("interpolation requires an autoencoder bundle")
        image_a, _ = self.sample(sample_id_a)
        image_b, _ = self.sample(sample_id_b)
        model = self.bundles[model_id].model
        model.eval()
        with torch.inference_mode():
            autoencoder = cast(EncoderDecoder, model)
            start = autoencoder.encode(image_a)
            end = autoencoder.encode(image_b)
            path, alphas = interpolate(start, end, steps)
            decoded = autoencoder.decode(path)
        return {
            "model_id": model_id,
            "model_version": self.version,
            "sample_id_a": sample_id_a,
            "sample_id_b": sample_id_b,
            "latent_distance": round(float(torch.linalg.vector_norm(end - start).item()), 6),
            "items": [
                {
                    "alpha": alpha,
                    "image": image_data_url(decoded[index]),
                }
                for index, alpha in enumerate(alphas)
            ],
            "warning": "Linear interpolation is not probabilistic sampling.",
        }

    def error_rows(
        self,
        model_id: str,
        class_id: int | None,
        limit: int,
    ) -> list[dict[str, object]]:
        self.require_model(model_id)
        if not 1 <= limit <= 30:
            raise ValueError("limit must be between 1 and 30")
        if model_id in ("mean-image", "pca"):
            prediction = self._baseline_reconstruct(model_id, self.images)
        else:
            model = self.bundles[model_id].model
            with torch.inference_mode():
                prediction = model(self.images)
        errors = per_sample_mse(self.images, prediction).tolist()
        rows = [
            {
                "sample_id": sample_id,
                "label": int(self.labels[index]),
                "class_name": CLASS_MAPPING[int(self.labels[index])],
                "mse": round(float(errors[index]), 7),
                "original": image_data_url(self.images[index]),
                "reconstruction": image_data_url(prediction[index]),
            }
            for index, sample_id in enumerate(self.sample_ids)
            if class_id is None or int(self.labels[index]) == class_id
        ]
        return sorted(
            rows,
            key=lambda item: float(cast(float, item["mse"])),
            reverse=True,
        )[:limit]


@lru_cache(maxsize=1)
def get_registry() -> LabRegistry:
    return LabRegistry()
