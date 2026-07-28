from __future__ import annotations

import argparse
import json
import sys
from typing import cast

from _common import PROJECT_ROOT, SRC_ROOT, load_prepared, loader, read_json, reset_directory

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from sequence_models.artifacts import save_bundle  # noqa: E402
from sequence_models.contracts import ModelConfig, ModelType, TrainingConfig  # noqa: E402
from sequence_models.evaluation import classification_metrics  # noqa: E402
from sequence_models.models import build_model, count_parameters  # noqa: E402
from sequence_models.training import evaluate, train  # noqa: E402


def run_training(model_name: str, version: str, force: bool) -> dict[str, object]:
    model_type = cast(ModelType, model_name)
    prepared = load_prepared()
    training_config = TrainingConfig()
    model_config = ModelConfig(model_type=model_type, hidden_size=64, dropout=0.2)
    train_loader = loader(
        prepared["train_values"],
        prepared["train_labels"],
        training_config.batch_size,
        True,
    )
    validation_loader = loader(
        prepared["validation_values"],
        prepared["validation_labels"],
        training_config.batch_size,
        False,
    )
    test_loader = loader(
        prepared["test_values"],
        prepared["test_labels"],
        training_config.batch_size,
        False,
    )
    model = build_model(model_config)
    training = train(model, train_loader, validation_loader, training_config)
    test_loss, truth, predictions = evaluate(model, test_loader)
    metrics = classification_metrics(truth, predictions).to_dict()
    metrics.update(
        {
            "test_loss": round(test_loss, 6),
            "validation_macro_f1": round(training.best_validation_macro_f1, 6),
            "best_epoch": training.best_epoch,
            "training_seconds": round(training.elapsed_seconds, 4),
            "parameters": count_parameters(model),
            "artifact_status": "official_uci_har",
        }
    )
    preparation = read_json(PROJECT_ROOT / "data" / "processed" / "preparation_manifest.json")
    bundle_path = PROJECT_ROOT / "artifacts" / "models" / model_name / version
    reset_directory(bundle_path, force)
    save_bundle(
        bundle_path,
        model,
        model_config,
        metrics,
        {
            "mean": prepared["mean"].tolist(),
            "std": prepared["std"].tolist(),
            "channels": preparation["channels"],
            "fitted_on": "official UCI HAR grouped training split only",
        },
        {
            "task": "many-to-one sequence classification",
            "model_id": model_name,
            "version": version,
            "dataset": "UCI HAR Smartphones",
            "dataset_status": "official",
            "dataset_provenance": preparation["dataset"],
            "input_shape": [None, 128, 9],
            "class_mapping": preparation["class_mapping"],
            "split_strategy": preparation["split_strategy"],
            "seed": training_config.seed,
            "device": "cpu",
            "selection_metric": "validation macro F1",
        },
    )
    (bundle_path / "training_history.json").write_text(
        json.dumps(training.history_dicts(), indent=2),
        encoding="utf-8",
    )
    return {"bundle": str(bundle_path), "metrics": metrics}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a recurrent UCI HAR classifier.")
    parser.add_argument("--model", choices=("rnn", "lstm", "gru"), required=True)
    parser.add_argument("--version", default="v1.1.0-uci")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run_training(args.model, args.version, args.force), indent=2))


if __name__ == "__main__":
    main()

