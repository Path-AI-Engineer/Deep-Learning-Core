"""Reproducible experiment runner and evidence generation."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from neural_network_foundations.contracts import ExperimentConfig
from neural_network_foundations.datasets import get_dataset
from neural_network_foundations.evaluation import (
    binary_accuracy,
    check_model_gradients,
    decision_boundary,
    diagnose,
)
from neural_network_foundations.models import MLP
from neural_network_foundations.models.pytorch_reference import compare_with_pytorch
from neural_network_foundations.serialization import save_checkpoint, write_json
from neural_network_foundations.training import train


@dataclass(frozen=True)
class ExperimentResult:
    run_id: str
    status: str
    dataset: dict[str, Any]
    initial_loss: float
    final_loss: float
    final_accuracy: float
    completed_epochs: int
    trace_path: str
    checkpoint_path: str
    summary_path: str
    gradient_check_passed: bool
    parity_passed: bool | None
    diagnostics: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _run_id(config: ExperimentConfig) -> str:
    encoded = json.dumps(config.to_dict(), sort_keys=True).encode("utf-8")
    digest = hashlib.sha256(encoded).hexdigest()[:8]
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    return f"{timestamp}-{digest}"


def _write_summary(path: Path, result: ExperimentResult) -> None:
    lines = [
        f"# Experiment {result.run_id}",
        "",
        f"- Status: **{result.status}**",
        f"- Dataset: **{result.dataset['name']}** ({result.dataset['samples']} samples)",
        f"- Initial loss: **{result.initial_loss:.8f}**",
        f"- Final loss: **{result.final_loss:.8f}**",
        f"- Final accuracy: **{result.final_accuracy:.2%}**",
        f"- Completed epochs: **{result.completed_epochs}**",
        f"- Gradient check: **{'passed' if result.gradient_check_passed else 'failed'}**",
        (
            "- PyTorch parity: "
            f"**{result.parity_passed if result.parity_passed is not None else 'not requested'}**"
        ),
        "",
        (
            "This run is pedagogical evidence on a bounded dataset, "
            "not a production generalization claim."
        ),
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_experiment(
    config: ExperimentConfig,
    *,
    artifact_root: str | Path = "artifacts/runs",
    run_id: str | None = None,
    include_parity: bool = True,
) -> ExperimentResult:
    identifier = run_id or _run_id(config)
    run_directory = Path(artifact_root) / identifier
    run_directory.mkdir(parents=True, exist_ok=False)
    dataset = get_dataset(config.dataset, seed=config.seed)
    model = MLP(config.network, seed=config.seed)
    initial_predictions = model.forward(dataset.features)
    initial_loss = model.calculate_loss(dataset.targets, loss_name=config.loss)
    initial_boundary = decision_boundary(
        model,
        dataset.features,
        resolution=config.grid_resolution,
    )
    gradient_model = MLP(config.network, seed=config.seed)
    gradient_report = check_model_gradients(
        gradient_model,
        dataset.features,
        dataset.targets,
        loss_name=config.loss,
    )
    history = train(
        model,
        dataset.features,
        dataset.targets,
        loss_name=config.loss,
        learning_rate=config.learning_rate,
        epochs=config.epochs,
    )
    final_predictions = model.forward(dataset.features)
    final_loss = model.calculate_loss(dataset.targets, loss_name=config.loss)
    model.backward(dataset.targets, loss_name=config.loss)
    trace = model.trace_sample(
        dataset=dataset.name,
        features=dataset.features,
        targets=dataset.targets,
        sample_index=min(1, dataset.features.shape[0] - 1),
        loss_name=config.loss,
        configuration=config.to_dict(),
    )
    final_boundary = decision_boundary(
        model,
        dataset.features,
        resolution=config.grid_resolution,
    )
    parity_report = None
    if include_parity:
        parity_report = compare_with_pytorch(
            model,
            dataset.features,
            dataset.targets,
            loss_name=config.loss,
            learning_rate=config.learning_rate,
        )
    trace_path = write_json(run_directory / "trace.json", trace.to_dict())
    write_json(run_directory / "config.json", config.to_dict())
    write_json(
        run_directory / "metrics.json",
        {
            "initial_predictions": initial_predictions.tolist(),
            "final_predictions": final_predictions.tolist(),
            "history": history.to_dict(),
            "initial_boundary": initial_boundary,
            "final_boundary": final_boundary,
            "gradient_check": gradient_report.to_dict(),
            "pytorch_parity": None if parity_report is None else parity_report.to_dict(),
        },
    )
    checkpoint_path, _ = save_checkpoint(
        model,
        run_directory / "checkpoint.npz",
        metadata={"run_id": identifier, "configuration": config.to_dict()},
    )
    diagnostic_values = [item.to_dict() for item in diagnose(model, history.loss)]
    summary_path = run_directory / "summary.md"
    result = ExperimentResult(
        run_id=identifier,
        status=history.status,
        dataset=dataset.metadata,
        initial_loss=initial_loss,
        final_loss=final_loss,
        final_accuracy=binary_accuracy(final_predictions, dataset.targets),
        completed_epochs=history.completed_epochs,
        trace_path=str(trace_path),
        checkpoint_path=str(checkpoint_path),
        summary_path=str(summary_path),
        gradient_check_passed=gradient_report.passed,
        parity_passed=None if parity_report is None else parity_report.passed,
        diagnostics=diagnostic_values,
    )
    _write_summary(summary_path, result)
    write_json(run_directory / "result.json", result.to_dict())
    return result
