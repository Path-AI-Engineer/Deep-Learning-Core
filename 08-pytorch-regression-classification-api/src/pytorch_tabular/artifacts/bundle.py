from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import torch

from pytorch_tabular.experiments import ExperimentResult

COMMON_FILES = (
    "model_state.pt",
    "model_config.json",
    "preprocessing.json",
    "feature_schema.json",
    "metrics.json",
    "training_history.json",
    "metadata.json",
)


@dataclass(frozen=True)
class BundleMetadata:
    schema_version: str
    model_version: str
    task: str
    dataset: str
    architecture: dict[str, object]
    feature_names: list[str]
    feature_schema: list[dict[str, object]]
    class_names: list[str]
    target_unit: str | None
    metrics: dict[str, object]
    baseline_metrics: dict[str, object]
    history: dict[str, list[float]]
    limitations: list[str]
    examples: list[dict[str, float]]
    source: str
    sample_count: int


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_bundle(
    result: ExperimentResult,
    destination: Path,
    model_version: str = "1.0.0",
    *,
    overwrite: bool = False,
) -> Path:
    if destination.exists():
        if not overwrite:
            raise FileExistsError(
                f"Bundle {destination} already exists. Approved versions are immutable."
            )
    destination.mkdir(parents=True, exist_ok=overwrite)
    torch.save(result.model.state_dict(), destination / "model_state.pt")
    hidden_units = cast(tuple[int, ...], getattr(result.model, "hidden_units"))
    dropout = cast(float, getattr(result.model, "dropout"))
    architecture: dict[str, Any] = {
        "input_features": len(result.data.feature_names),
        "hidden_units": list(hidden_units),
        "dropout": dropout,
        "output_units": (
            cast(int, getattr(result.model, "class_count"))
            if result.task == "classification"
            else 1
        ),
    }
    preprocessing = {
        "type": "StandardScaler",
        "mean": result.data.scaler.mean_.tolist(),
        "scale": result.data.scaler.scale_.tolist(),
        "variance": result.data.scaler.var_.tolist(),
        "feature_count": int(result.data.scaler.n_features_in_),
        "samples_seen": int(result.data.scaler.n_samples_seen_),
        "fit_scope": "training split only",
    }
    metadata = BundleMetadata(
        schema_version="1.0",
        model_version=model_version,
        task=result.task,
        dataset=result.data.dataset_name,
        architecture=architecture,
        feature_names=result.data.feature_names,
        feature_schema=[asdict(spec) for spec in result.data.feature_specs],
        class_names=result.data.class_names,
        target_unit=result.data.target_unit,
        metrics=result.metrics,
        baseline_metrics=result.baseline_metrics,
        history=result.history,
        limitations=[
            "Predictions are estimates from a bounded educational dataset.",
            "Inputs outside observed training ranges may be unreliable.",
            "Probability is not certainty and does not establish causality.",
        ],
        examples=[
            {
                name: float(spec.example)
                for name, spec in zip(result.data.feature_names, result.data.feature_specs)
            }
        ],
        source=result.data.source,
        sample_count=(
            len(result.data.x_train)
            + len(result.data.x_validation)
            + len(result.data.x_test)
        ),
    )
    (destination / "metadata.json").write_text(
        json.dumps(asdict(metadata), indent=2),
        encoding="utf-8",
    )
    (destination / "model_config.json").write_text(
        json.dumps(architecture, indent=2),
        encoding="utf-8",
    )
    (destination / "preprocessing.json").write_text(
        json.dumps(preprocessing, indent=2),
        encoding="utf-8",
    )
    (destination / "feature_schema.json").write_text(
        json.dumps(asdict(metadata)["feature_schema"], indent=2),
        encoding="utf-8",
    )
    (destination / "metrics.json").write_text(
        json.dumps(
            {
                "test": result.metrics,
                "baseline": result.baseline_metrics,
                "primary_metric": "mae" if result.task == "regression" else "macro_f1",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (destination / "training_history.json").write_text(
        json.dumps(result.history, indent=2),
        encoding="utf-8",
    )
    files = list(COMMON_FILES)
    if result.task == "classification":
        class_mapping = {
            str(index): name for index, name in enumerate(result.data.class_names)
        }
        (destination / "class_mapping.json").write_text(
            json.dumps(class_mapping, indent=2),
            encoding="utf-8",
        )
        (destination / "confusion_matrix.json").write_text(
            json.dumps(result.metrics["confusion_matrix"], indent=2),
            encoding="utf-8",
        )
        files.extend(("class_mapping.json", "confusion_matrix.json"))
    manifest = {
        "schema_version": "1.0",
        "task": result.task,
        "model_version": model_version,
        "files": {name: _sha256(destination / name) for name in files},
    }
    (destination / "manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    return destination


def validate_bundle(bundle: Path) -> dict[str, Any]:
    manifest_path = bundle / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError("Incomplete model bundle. Missing: manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    required = list(COMMON_FILES)
    if manifest.get("task") == "classification":
        required.extend(("class_mapping.json", "confusion_matrix.json"))
    missing = [name for name in (*required, "manifest.json") if not (bundle / name).is_file()]
    if missing:
        raise ValueError(f"Incomplete model bundle. Missing: {', '.join(missing)}")
    if set(manifest.get("files", {})) != set(required):
        raise ValueError("Manifest file inventory does not match the bundle contract.")
    for name, expected in manifest["files"].items():
        if _sha256(bundle / name) != expected:
            raise ValueError(f"Hash mismatch for bundle file: {name}")
    metadata = json.loads((bundle / "metadata.json").read_text(encoding="utf-8"))
    if metadata["feature_names"] != [
        feature["name"] for feature in metadata["feature_schema"]
    ]:
        raise ValueError("Feature schema order does not match feature_names.")
    return metadata
