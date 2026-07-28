from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from sequence_models.artifacts import save_bundle  # noqa: E402
from sequence_models.contracts import (  # noqa: E402
    CHANNELS,
    CLASS_MAPPING,
    ModelConfig,
    TrainingConfig,
)
from sequence_models.data import SequenceRecord, build_demo_records  # noqa: E402
from sequence_models.evaluation import classification_metrics  # noqa: E402
from sequence_models.models import build_model, count_parameters  # noqa: E402
from sequence_models.training import evaluate, train  # noqa: E402


def _loader(
    indices: list[int],
    records: list[SequenceRecord],
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    values = np.stack([records[index].values for index in indices]).astype(np.float32)
    labels = np.asarray([records[index].label for index in indices], dtype=np.int64)
    dataset = TensorDataset(torch.from_numpy(values), torch.from_numpy(labels))
    generator = torch.Generator().manual_seed(42)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, generator=generator)


def build_fixture_bundles(force: bool = False) -> dict[str, object]:
    records = build_demo_records(20)
    train_indices = [
        index for index, record in enumerate(records) if int(record.sample_id[-2:]) < 14
    ]
    validation_indices = [
        index for index, record in enumerate(records) if 14 <= int(record.sample_id[-2:]) < 17
    ]
    test_indices = [
        index for index, record in enumerate(records) if int(record.sample_id[-2:]) >= 17
    ]
    train_loader = _loader(train_indices, records, 32, True)
    validation_loader = _loader(validation_indices, records, 32, False)
    test_loader = _loader(test_indices, records, 32, False)
    comparison: list[dict[str, object]] = []
    for model_type in ("rnn", "lstm", "gru"):
        torch.manual_seed(42)
        config = ModelConfig(model_type=model_type, hidden_size=24, dropout=0.1)
        model = build_model(config)
        result = train(
            model,
            train_loader,
            validation_loader,
            TrainingConfig(
                seed=42,
                epochs=14,
                batch_size=32,
                learning_rate=0.006,
                gradient_clip_norm=1.0,
                early_stopping_patience=4,
            ),
        )
        test_loss, truth, predictions = evaluate(model, test_loader)
        metrics = classification_metrics(truth, predictions).to_dict()
        metrics.update(
            {
                "test_loss": round(test_loss, 6),
                "validation_macro_f1": round(result.best_validation_macro_f1, 6),
                "best_epoch": result.best_epoch,
                "training_seconds": round(result.elapsed_seconds, 4),
                "parameters": count_parameters(model),
                "artifact_status": "educational_fixture",
            }
        )
        bundle_path = PROJECT_ROOT / "artifacts" / "models" / model_type / "v1.0.0"
        if bundle_path.exists():
            if not force:
                raise FileExistsError(f"{bundle_path} already exists; use --force")
            shutil.rmtree(bundle_path)
        save_bundle(
            bundle_path,
            model,
            config,
            metrics,
            {
                "mean": [0.0] * 9,
                "std": [1.0] * 9,
                "channels": list(CHANNELS),
                "fitted_on": "deterministic educational fixture training split",
            },
            {
                "task": "many-to-one sequence classification",
                "model_id": model_type,
                "version": "v1.0.0",
                "dataset": "deterministic HAR-shaped educational fixture",
                "dataset_status": "fixture-not-uci",
                "official_dataset_target": "UCI HAR Smartphones",
                "input_shape": [None, 128, 9],
                "class_mapping": CLASS_MAPPING,
                "seed": 42,
                "device": "cpu",
                "selection_metric": "validation macro F1",
                "warning": (
                    "Replace this fixture bundle with an official UCI-trained bundle for claims."
                ),
            },
        )
        (bundle_path / "training_history.json").write_text(
            json.dumps(result.history_dicts(), indent=2), encoding="utf-8"
        )
        comparison.append({"model_id": model_type, **metrics})
    comparison_root = PROJECT_ROOT / "artifacts" / "comparisons" / "v1.0.0"
    comparison_root.mkdir(parents=True, exist_ok=True)
    approved = max(comparison, key=lambda item: float(item["validation_macro_f1"]))
    payload = {
        "comparison_version": "v1.0.0",
        "data_mode": "educational_fixture",
        "selection_metric": "validation_macro_f1",
        "approved_model": approved["model_id"],
        "models": comparison,
        "warning": "These measurements validate the system; they are not UCI HAR benchmark claims.",
    }
    (comparison_root / "model_comparison.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    arguments = parser.parse_args()
    print(json.dumps(build_fixture_bundles(arguments.force), indent=2))


if __name__ == "__main__":
    main()
