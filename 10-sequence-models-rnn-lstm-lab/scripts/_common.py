from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def prepared_path() -> Path:
    path = PROJECT_ROOT / "data" / "processed" / "uci_har_prepared.npz"
    if not path.is_file():
        raise FileNotFoundError("Prepared data is missing. Run scripts/prepare_data.py first.")
    return path


def load_prepared() -> dict[str, np.ndarray]:
    with np.load(prepared_path()) as archive:
        return {name: archive[name] for name in archive.files}


def loader(
    values: np.ndarray,
    labels: np.ndarray,
    batch_size: int,
    shuffle: bool,
    seed: int = 42,
) -> DataLoader:
    dataset = TensorDataset(
        torch.from_numpy(values.astype(np.float32)),
        torch.from_numpy(labels.astype(np.int64)),
    )
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
    )


def reset_directory(path: Path, force: bool) -> None:
    if path.exists():
        if not force:
            raise FileExistsError(f"{path} already exists; pass --force to replace it")
        shutil.rmtree(path)

