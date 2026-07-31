from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from transformer_lab.contracts import ModelConfig
from transformer_lab.models import SequenceTransformer

JSON_FILES = (
    "model_config.json",
    "vocabulary.json",
    "task_config.json",
    "preprocessing.json",
    "decoding_config.json",
    "data_manifest.json",
    "split_manifest.json",
    "metrics.json",
    "per_task_metrics.json",
    "per_length_metrics.json",
    "latency.json",
    "training_history.json",
)


@dataclass(frozen=True)
class Bundle:
    model: SequenceTransformer
    manifest: dict[str, Any]
    files: dict[str, Any]


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
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
    model: SequenceTransformer,
    *,
    documents: dict[str, Any],
    metadata: dict[str, Any],
) -> Path:
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"Bundle already exists: {destination}.")
    missing = set(JSON_FILES) - set(documents)
    if missing:
        raise ValueError(f"Bundle documents are missing: {sorted(missing)}.")
    destination.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), destination / "model_state.pt")
    for filename in JSON_FILES:
        _write_json(destination / filename, documents[filename])
    hashes = {
        path.name: _sha256(path)
        for path in sorted(destination.iterdir())
        if path.is_file()
    }
    manifest = {
        **metadata,
        "required_files": ["model_state.pt", *JSON_FILES],
        "hashes": hashes,
        "expected_device": "cpu",
    }
    _write_json(destination / "manifest.json", manifest)
    return destination


def load_bundle(path: Path) -> Bundle:
    required = {"model_state.pt", "manifest.json", *JSON_FILES}
    present = {item.name for item in path.iterdir()} if path.exists() else set()
    if missing := required - present:
        raise ValueError(f"Incomplete Transformer bundle: {sorted(missing)}.")
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    for filename, expected_hash in manifest.get("hashes", {}).items():
        if _sha256(path / filename) != expected_hash:
            raise ValueError(f"Bundle hash mismatch: {filename}.")
    files = {
        filename: json.loads((path / filename).read_text(encoding="utf-8"))
        for filename in JSON_FILES
    }
    config = ModelConfig(**files["model_config.json"])
    model = SequenceTransformer(config)
    state = torch.load(
        path / "model_state.pt",
        map_location="cpu",
        weights_only=True,
    )
    model.load_state_dict(state)
    model.eval()
    return Bundle(model=model, manifest=manifest, files=files)

