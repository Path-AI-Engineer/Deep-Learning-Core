from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from cnn_foundations.models.cnn import FashionCNN, model_from_config

JSON_FILES = (
    "model_config.json",
    "preprocessing.json",
    "class_mapping.json",
    "metrics.json",
    "per_class_metrics.json",
    "confusion_matrix.json",
    "training_history.json",
    "comparison_with_mlp.json",
    "error_analysis.json",
    "split_manifest.json",
)


@dataclass(frozen=True)
class BundleContents:
    model: FashionCNN
    metadata: dict[str, Any]
    files: dict[str, Any]


def _json_write(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_bundle(
    destination: Path,
    model: FashionCNN,
    *,
    metadata: dict[str, Any],
    documents: dict[str, Any],
) -> Path:
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"bundle version already exists: {destination}")
    missing = set(JSON_FILES) - set(documents)
    if missing:
        raise ValueError(f"bundle documents are missing: {sorted(missing)}")
    destination.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), destination / "model_state.pt")
    for filename in JSON_FILES:
        _json_write(destination / filename, documents[filename])
    hashes = {
        path.name: _sha256(path)
        for path in sorted(destination.iterdir())
        if path.is_file() and path.name != "manifest.json"
    }
    manifest = {**metadata, "hashes": hashes, "compatibility": {"device": "cpu"}}
    _json_write(destination / "manifest.json", manifest)
    return destination


def load_bundle(path: Path) -> BundleContents:
    required = {"model_state.pt", "manifest.json", *JSON_FILES}
    present = {item.name for item in path.iterdir()} if path.exists() else set()
    if missing := required - present:
        raise ValueError(f"incomplete CNN bundle: {sorted(missing)}")
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    hashes = manifest.get("hashes", {})
    for filename, expected in hashes.items():
        if _sha256(path / filename) != expected:
            raise ValueError(f"bundle hash mismatch: {filename}")
    files = {
        filename: json.loads((path / filename).read_text(encoding="utf-8"))
        for filename in JSON_FILES
    }
    model = model_from_config(files["model_config.json"])
    state = torch.load(path / "model_state.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return BundleContents(model=model, metadata=manifest, files=files)

