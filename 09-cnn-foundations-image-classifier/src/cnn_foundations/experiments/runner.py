from __future__ import annotations

import json
import subprocess
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from cnn_foundations.contracts.config import ModelConfig, load_experiment_config
from cnn_foundations.data.fashion_mnist import build_dataloaders
from cnn_foundations.evaluation.error_analysis import build_error_records
from cnn_foundations.evaluation.metrics import evaluate_predictions
from cnn_foundations.models.cnn import FashionCNN
from cnn_foundations.models.mlp import FashionMLP
from cnn_foundations.training.engine import fit
from cnn_foundations.utils.runtime import environment_snapshot, resolve_device, seed_everything


def _commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _build_model(config: ModelConfig) -> nn.Module:
    if config.kind == "cnn":
        assert config.conv_channels is not None
        return FashionCNN(
            channels=config.conv_channels,
            hidden_features=config.hidden_features,
            dropout=config.dropout,
            batch_norm=config.batch_norm,
        )
    return FashionMLP(
        hidden_features=config.hidden_features,
        dropout=config.dropout,
    )


class ExperimentRunner:
    def __init__(self, output_root: Path = Path("artifacts/runs")) -> None:
        self.output_root = output_root

    def run(self, config_path: Path, *, evaluate_test: bool = False) -> Path:
        experiment, dataset, model_config = load_experiment_config(config_path)
        seed_everything(experiment.seed)
        device = resolve_device(experiment.device)
        loaders = build_dataloaders(
            dataset,
            experiment.augmentation,
            batch_size=experiment.batch_size,
        )
        model = _build_model(model_config)
        result = fit(
            model,
            loaders.train,
            loaders.validation,
            device=device,
            epochs=experiment.epochs,
            patience=experiment.patience,
            learning_rate=experiment.learning_rate,
            weight_decay=experiment.weight_decay,
        )
        run_id = (
            datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
            + "-"
            + uuid.uuid4().hex[:8]
        )
        destination = self.output_root / experiment.name / run_id
        destination.mkdir(parents=True, exist_ok=False)
        torch.save(result.best_state, destination / "best_state.pt")
        payload: dict[str, Any] = {
            "run_id": run_id,
            "experiment": experiment.model_dump(mode="json"),
            "dataset": dataset.model_dump(mode="json"),
            "model": model_config.model_dump(mode="json"),
            "best_epoch": result.best_epoch,
            "history": list(result.history),
            "split": loaders.split.as_dict(),
            "environment": environment_snapshot(),
            "commit": _commit(),
            "test_evaluated": False,
        }
        if evaluate_test:
            model.load_state_dict(result.best_state)
            model.to(device).eval()
            expected: list[int] = []
            predicted: list[int] = []
            probabilities: list[list[float]] = []
            with torch.inference_mode():
                for inputs, targets in loaders.test:
                    logits = model(inputs.to(device))
                    scores = torch.softmax(logits, dim=1).cpu()
                    expected.extend(int(value) for value in targets)
                    predicted.extend(int(value) for value in logits.argmax(dim=1).cpu())
                    probabilities.extend(scores.tolist())
            payload["test"] = evaluate_predictions(
                np.asarray(expected, dtype=np.int64),
                np.asarray(predicted, dtype=np.int64),
            ).as_dict()
            payload["error_analysis"] = {
                "errors": build_error_records(
                    expected,
                    np.asarray(probabilities, dtype=np.float64),
                    [f"test-{index:05d}" for index in range(len(expected))],
                    limit=100,
                )
            }
            payload["test_evaluated"] = True
        (destination / "run.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return destination
