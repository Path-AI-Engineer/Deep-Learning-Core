"""Small deterministic datasets with metadata for the visual explainer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class DatasetBundle:
    name: str
    description: str
    purpose: str
    features: FloatArray
    targets: FloatArray
    seed: int

    def __post_init__(self) -> None:
        x = np.asarray(self.features, dtype=np.float64)
        y = np.asarray(self.targets, dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != 2:
            raise ValueError("features must have shape (samples, 2).")
        if y.ndim != 2 or y.shape != (x.shape[0], 1):
            raise ValueError("targets must have shape (samples, 1).")
        if x.shape[0] == 0 or x.shape[0] > 1000:
            raise ValueError("datasets must contain between 1 and 1000 samples.")
        if not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError("dataset values must be finite.")
        if not np.isin(y, [0.0, 1.0]).all():
            raise ValueError("targets must contain binary labels.")
        object.__setattr__(self, "features", x)
        object.__setattr__(self, "targets", y)

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "purpose": self.purpose,
            "feature_shape": list(self.features.shape),
            "target_shape": list(self.targets.shape),
            "problem_type": "binary_classification",
            "samples": int(self.features.shape[0]),
            "seed": self.seed,
            "feature_range": [
                float(np.min(self.features)),
                float(np.max(self.features)),
            ],
        }

    def to_dict(self, *, include_points: bool = True) -> dict[str, Any]:
        value = self.metadata
        if include_points:
            value = {
                **value,
                "features": self.features.tolist(),
                "targets": self.targets.reshape(-1).tolist(),
            }
        return value


def _truth_table(name: str, labels: list[int], purpose: str) -> DatasetBundle:
    features = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float64)
    targets = np.asarray(labels, dtype=np.float64).reshape(-1, 1)
    return DatasetBundle(
        name=name,
        description=f"Binary {name.upper()} truth table.",
        purpose=purpose,
        features=features,
        targets=targets,
        seed=0,
    )


def _circles(seed: int = 17, samples: int = 160) -> DatasetBundle:
    if samples < 20 or samples > 1000:
        raise ValueError("circles samples must be between 20 and 1000.")
    rng = np.random.default_rng(seed)
    outer_count = samples // 2
    inner_count = samples - outer_count
    outer_angle = rng.uniform(0.0, 2.0 * np.pi, outer_count)
    inner_angle = rng.uniform(0.0, 2.0 * np.pi, inner_count)
    outer_radius = rng.normal(1.0, 0.06, outer_count)
    inner_radius = rng.normal(0.45, 0.05, inner_count)
    outer = np.column_stack(
        (outer_radius * np.cos(outer_angle), outer_radius * np.sin(outer_angle))
    )
    inner = np.column_stack(
        (inner_radius * np.cos(inner_angle), inner_radius * np.sin(inner_angle))
    )
    features = np.vstack((outer, inner)).astype(np.float64)
    targets = np.vstack((np.zeros((outer_count, 1)), np.ones((inner_count, 1))))
    order = rng.permutation(samples)
    return DatasetBundle(
        name="circles",
        description="Noisy concentric circles with two nonlinear classes.",
        purpose="Observe a curved decision boundary learned by a small MLP.",
        features=features[order],
        targets=targets[order],
        seed=seed,
    )


def get_dataset(name: str, *, seed: int | None = None, samples: int = 160) -> DatasetBundle:
    normalized = name.strip().lower()
    if normalized == "and":
        return _truth_table("and", [0, 0, 0, 1], "Reference a linearly separable logical rule.")
    if normalized == "or":
        return _truth_table("or", [0, 1, 1, 1], "Reference a second linearly separable rule.")
    if normalized == "xor":
        return _truth_table(
            "xor",
            [0, 1, 1, 0],
            "Show why a nonlinear hidden representation is required.",
        )
    if normalized == "circles":
        return _circles(seed=17 if seed is None else seed, samples=samples)
    raise KeyError(f"Unknown dataset: {name}.")


def list_datasets() -> list[dict[str, Any]]:
    return [
        get_dataset(name).to_dict(include_points=False) for name in ("and", "or", "xor", "circles")
    ]
