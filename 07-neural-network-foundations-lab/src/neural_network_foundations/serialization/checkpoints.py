"""NumPy checkpoint persistence with explicit metadata."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from neural_network_foundations.models import MLP
from neural_network_foundations.serialization.json_artifacts import read_json, write_json

FloatArray = NDArray[np.float64]


def save_checkpoint(
    model: MLP,
    path: str | Path,
    *,
    metadata: dict[str, Any],
) -> tuple[Path, Path]:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as checkpoint_file:
        # NumPy's stub interprets variadic named arrays as the `allow_pickle`
        # flag even though the runtime API accepts them as archive members.
        np.savez(checkpoint_file, **model.parameters())  # type: ignore[arg-type]
    metadata_path = destination.with_suffix(".json")
    write_json(
        metadata_path,
        {
            "schema_version": "1.0",
            "checkpoint": destination.name,
            "parameters": {
                name: {"shape": list(value.shape), "dtype": str(value.dtype)}
                for name, value in model.parameters().items()
            },
            "metadata": metadata,
        },
    )
    return destination, metadata_path


def load_checkpoint(path: str | Path) -> tuple[dict[str, FloatArray], dict[str, Any]]:
    source = Path(path)
    metadata_path = source.with_suffix(".json")
    if not source.exists() or not metadata_path.exists():
        raise FileNotFoundError("Checkpoint data and metadata must both exist.")
    metadata = read_json(metadata_path)
    declared = metadata.get("parameters")
    if not isinstance(declared, dict):
        raise ValueError("Checkpoint metadata does not declare parameters.")
    with np.load(source, allow_pickle=False) as archive:
        values = {name: np.asarray(archive[name], dtype=np.float64) for name in archive.files}
    if set(values) != set(declared):
        raise ValueError("Checkpoint parameter keys do not match metadata.")
    for name, value in values.items():
        if list(value.shape) != declared[name]["shape"]:
            raise ValueError(f"Checkpoint shape mismatch for {name}.")
        if not np.isfinite(value).all():
            raise ValueError(f"Checkpoint contains non-finite values for {name}.")
    return values, metadata
