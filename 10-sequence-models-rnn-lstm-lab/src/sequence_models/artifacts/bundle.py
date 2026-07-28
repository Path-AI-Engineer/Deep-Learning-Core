from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from sequence_models.contracts import ModelConfig
from sequence_models.models import build_model


class BundleError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class ModelBundle:
    model: nn.Module
    model_id: str
    version: str
    manifest: dict[str, Any]
    metrics: dict[str, Any]
    preprocessing: dict[str, Any]


def _json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise BundleError(f"missing required bundle file: {path.name}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise BundleError(f"{path.name} must contain a JSON object")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_bundle(
    directory: str | Path,
    model: nn.Module,
    config: ModelConfig,
    metrics: dict[str, Any],
    preprocessing: dict[str, Any],
    manifest: dict[str, Any],
) -> Path:
    bundle_path = Path(directory)
    bundle_path.mkdir(parents=True, exist_ok=False)
    model_state = bundle_path / "model_state.pt"
    torch.save(model.state_dict(), model_state)
    model_config = {
        "model_type": config.model_type,
        "input_size": config.input_size,
        "hidden_size": config.hidden_size,
        "num_layers": config.num_layers,
        "num_classes": config.num_classes,
        "dropout": config.dropout,
        "batch_first": config.batch_first,
    }
    (bundle_path / "model_config.json").write_text(
        json.dumps(model_config, indent=2), encoding="utf-8"
    )
    (bundle_path / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (bundle_path / "preprocessing.json").write_text(
        json.dumps(preprocessing, indent=2), encoding="utf-8"
    )
    completed_manifest = dict(manifest)
    completed_manifest["state_sha256"] = _sha256(model_state)
    (bundle_path / "manifest.json").write_text(
        json.dumps(completed_manifest, indent=2), encoding="utf-8"
    )
    return bundle_path


def load_bundle(directory: str | Path) -> ModelBundle:
    bundle_path = Path(directory)
    config_payload = _json(bundle_path / "model_config.json")
    metrics = _json(bundle_path / "metrics.json")
    preprocessing = _json(bundle_path / "preprocessing.json")
    manifest = _json(bundle_path / "manifest.json")
    state_path = bundle_path / "model_state.pt"
    if not state_path.is_file():
        raise BundleError("missing required bundle file: model_state.pt")
    if manifest.get("state_sha256") != _sha256(state_path):
        raise BundleError("model_state.pt hash does not match the manifest")
    try:
        config = ModelConfig(**config_payload)
    except TypeError as exc:
        raise BundleError("model_config.json is incompatible") from exc
    model = build_model(config)
    state = torch.load(state_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return ModelBundle(
        model=model,
        model_id=str(manifest["model_id"]),
        version=str(manifest["version"]),
        manifest=manifest,
        metrics=metrics,
        preprocessing=preprocessing,
    )
