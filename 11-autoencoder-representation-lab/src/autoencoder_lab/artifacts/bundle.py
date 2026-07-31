from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from autoencoder_lab.contracts import ModelConfig
from autoencoder_lab.models import build_model


class BundleError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class AutoencoderBundle:
    model: nn.Module
    model_id: str
    version: str
    manifest: dict[str, Any]
    metrics: dict[str, Any]


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise BundleError(f"missing bundle file: {path.name}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise BundleError(f"{path.name} must contain an object")
    return value


def save_bundle(
    directory: Path,
    model: nn.Module,
    config: ModelConfig,
    metrics: dict[str, Any],
    history: list[dict[str, int | float]],
    manifest: dict[str, Any],
) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    state_path = directory / "model_state.pt"
    torch.save(model.state_dict(), state_path)
    files: dict[str, object] = {
        "model_config.json": {
            "model_type": config.model_type,
            "latent_dim": config.latent_dim,
            "input_shape": list(config.input_shape),
        },
        "preprocessing.json": {
            "dtype": "float32",
            "pixel_range": [0.0, 1.0],
            "shape": [1, 28, 28],
        },
        "metrics.json": metrics,
        "reconstruction_metrics.json": metrics["reconstruction"],
        "representation_metrics.json": metrics["representation"],
        "robustness_metrics.json": metrics["robustness"],
        "training_history.json": history,
    }
    for name, payload in files.items():
        (directory / name).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    completed = dict(manifest)
    completed["state_sha256"] = _hash(state_path)
    completed["required_files"] = ["model_state.pt", *files.keys(), "manifest.json"]
    (directory / "manifest.json").write_text(
        json.dumps(completed, indent=2),
        encoding="utf-8",
    )
    return directory


def load_bundle(directory: Path) -> AutoencoderBundle:
    config_value = _json(directory / "model_config.json")
    metrics = _json(directory / "metrics.json")
    manifest = _json(directory / "manifest.json")
    state_path = directory / "model_state.pt"
    if not state_path.is_file() or _hash(state_path) != manifest.get("state_sha256"):
        raise BundleError("model state is missing or its hash is invalid")
    try:
        input_shape = tuple(int(item) for item in config_value["input_shape"])
        config = ModelConfig(
            model_type=config_value["model_type"],
            latent_dim=int(config_value["latent_dim"]),
            input_shape=input_shape,  # type: ignore[arg-type]
        )
    except (KeyError, TypeError, ValueError) as error:
        raise BundleError("model configuration is incompatible") from error
    model = build_model(config)
    state = torch.load(state_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return AutoencoderBundle(
        model=model,
        model_id=str(manifest["model_id"]),
        version=str(manifest["model_version"]),
        manifest=manifest,
        metrics=metrics,
    )
