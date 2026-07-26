from __future__ import annotations

import json
import platform
import subprocess
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import sklearn
import torch
from sklearn.dummy import DummyClassifier, DummyRegressor

from pytorch_tabular.contracts import ExperimentConfig
from pytorch_tabular.data import PreparedTaskData, load_task_data
from pytorch_tabular.evaluation import classification_metrics, regression_metrics
from pytorch_tabular.models import ClassificationMLP, RegressionMLP
from pytorch_tabular.training import fit
from pytorch_tabular.utils import seed_everything


@dataclass
class ExperimentResult:
    run_id: str
    run_directory: Path
    task: str
    metrics: dict[str, Any]
    baseline_metrics: dict[str, Any]
    history: dict[str, list[float]]
    best_epoch: int
    model: torch.nn.Module
    data: PreparedTaskData
    config: ExperimentConfig


def _git_revision(root: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=root,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _predict(model: torch.nn.Module, features: np.ndarray, task: str) -> np.ndarray:
    model.eval()
    with torch.inference_mode():
        output = model(torch.as_tensor(features, dtype=torch.float32))
        if task == "classification":
            output = torch.softmax(output, dim=1)
    return output.cpu().numpy()


def run_experiment(
    config: ExperimentConfig,
    root: Path,
    data: PreparedTaskData | None = None,
) -> ExperimentResult:
    seed_everything(config.seed)
    data = data or load_task_data(config.task, config.seed)
    loaders = data.loaders(config.batch_size)
    if config.task == "regression":
        model: torch.nn.Module = RegressionMLP(len(data.feature_names))
        baseline = DummyRegressor(strategy="mean").fit(data.x_train, data.y_train)
        baseline_metrics: dict[str, Any] = regression_metrics(
            data.y_test, baseline.predict(data.x_test)
        )
    else:
        model = ClassificationMLP(len(data.feature_names), len(data.class_names))
        baseline = DummyClassifier(strategy="prior").fit(data.x_train, data.y_train)
        baseline_probabilities = baseline.predict_proba(data.x_test)
        baseline_metrics = classification_metrics(data.y_test, baseline_probabilities)

    fit_result = fit(model, loaders.train, loaders.validation, config)
    prediction = _predict(model, data.x_test, config.task)
    metrics: dict[str, Any] = (
        regression_metrics(data.y_test, prediction)
        if config.task == "regression"
        else classification_metrics(data.y_test, prediction)
    )
    run_id = f"{datetime.now(UTC).strftime('%Y%m%dT%H%M%S')}-{uuid4().hex[:8]}"
    run_directory = root / "artifacts" / "runs" / config.task / run_id
    run_directory.mkdir(parents=True, exist_ok=False)
    evidence = {
        "run_id": run_id,
        "created_at": datetime.now(UTC).isoformat(),
        "task": config.task,
        "config": config.to_dict(),
        "best_epoch": fit_result.best_epoch,
        "best_validation_loss": fit_result.best_validation_loss,
        "stopped_early": fit_result.stopped_early,
        "metrics": metrics,
        "baseline_metrics": baseline_metrics,
        "history": fit_result.history.to_dict(),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "sklearn": sklearn.__version__,
            "device": "cpu",
            "git_revision": _git_revision(root),
        },
    }
    (run_directory / "run.json").write_text(
        json.dumps(evidence, indent=2),
        encoding="utf-8",
    )
    torch.save(model.state_dict(), run_directory / "best-state-dict.pt")
    return ExperimentResult(
        run_id=run_id,
        run_directory=run_directory,
        task=config.task,
        metrics=metrics,
        baseline_metrics=baseline_metrics,
        history=fit_result.history.to_dict(),
        best_epoch=fit_result.best_epoch,
        model=model,
        data=data,
        config=config,
    )
