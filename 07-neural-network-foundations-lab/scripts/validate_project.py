"""Run mathematical, training, persistence and PyTorch acceptance checks."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from uuid import uuid4

from jsonschema import Draft202012Validator

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from neural_network_foundations.contracts import ExperimentConfig
from neural_network_foundations.datasets import get_dataset
from neural_network_foundations.evaluation import (
    binary_accuracy,
    check_model_gradients,
    decision_boundary,
)
from neural_network_foundations.models import MLP
from neural_network_foundations.models.pytorch_reference import compare_with_pytorch
from neural_network_foundations.serialization import (
    load_checkpoint,
    render_boundary_svg,
    save_checkpoint,
    write_json,
)
from neural_network_foundations.training import train


def main() -> None:
    config = ExperimentConfig()
    dataset = get_dataset("xor")
    model = MLP(config.network, seed=config.seed)
    initial_boundary = decision_boundary(
        model,
        dataset.features,
        resolution=config.grid_resolution,
    )
    gradient_report = check_model_gradients(
        model,
        dataset.features,
        dataset.targets,
        loss_name=config.loss,
    )
    if not gradient_report.passed:
        raise SystemExit("Gradient checking failed.")
    history = train(
        model,
        dataset.features,
        dataset.targets,
        loss_name=config.loss,
        learning_rate=config.learning_rate,
        epochs=config.epochs,
    )
    predictions = model.forward(dataset.features)
    accuracy = binary_accuracy(predictions, dataset.targets)
    if accuracy != 1.0 or history.loss[-1] >= 0.02:
        raise SystemExit("The approved XOR experiment did not satisfy closure criteria.")
    model.backward(dataset.targets, loss_name=config.loss)
    trace = model.trace_sample(
        dataset="xor",
        features=dataset.features,
        targets=dataset.targets,
        sample_index=1,
        loss_name=config.loss,
        configuration=config.to_dict(),
    ).to_dict()
    json.dumps(trace, allow_nan=False)
    trace_schema = json.loads(Path("contracts/trace-v1.schema.json").read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(trace_schema)
    Draft202012Validator(trace_schema).validate(trace)
    parity = compare_with_pytorch(
        model,
        dataset.features,
        dataset.targets,
        loss_name=config.loss,
        learning_rate=config.learning_rate,
    )
    if not parity.passed:
        raise SystemExit("PyTorch parity failed.")

    runtime = Path(".runtime/validation") / uuid4().hex
    checkpoint, _ = save_checkpoint(
        model,
        runtime / "xor.npz",
        metadata={"configuration": config.to_dict()},
    )
    restored_values, _ = load_checkpoint(checkpoint)
    restored = MLP(config.network, seed=999)
    restored.load_parameters(restored_values)
    if not (restored.forward(dataset.features) == predictions).all():
        raise SystemExit("Checkpoint reconstruction changed predictions.")

    final_boundary = decision_boundary(
        model,
        dataset.features,
        resolution=config.grid_resolution,
    )
    render_boundary_svg(
        "artifacts/figures/demo-xor-before.svg",
        title="XOR decision boundary · before training",
        boundary=initial_boundary,
        features=dataset.features,
        targets=dataset.targets,
    )
    render_boundary_svg(
        "artifacts/figures/demo-xor-after.svg",
        title="XOR decision boundary · after training",
        boundary=final_boundary,
        features=dataset.features,
        targets=dataset.targets,
    )

    circles_config = ExperimentConfig.from_dict(
        {
            "dataset": "circles",
            "network": {
                "input_features": 2,
                "hidden_units": 8,
                "output_units": 1,
                "hidden_activation": "tanh",
                "output_activation": "sigmoid",
                "initialization": "xavier",
            },
            "loss": "binary_cross_entropy",
            "learning_rate": 0.2,
            "epochs": 3000,
            "seed": 17,
            "grid_resolution": 50,
        }
    )
    circles = get_dataset("circles")
    circles_model = MLP(circles_config.network, seed=circles_config.seed)
    circles_initial_boundary = decision_boundary(
        circles_model,
        circles.features,
        resolution=circles_config.grid_resolution,
    )
    circles_history = train(
        circles_model,
        circles.features,
        circles.targets,
        loss_name=circles_config.loss,
        learning_rate=circles_config.learning_rate,
        epochs=circles_config.epochs,
    )
    circles_accuracy = binary_accuracy(
        circles_model.forward(circles.features),
        circles.targets,
    )
    if circles_accuracy < 0.95 or circles_history.loss[-1] >= 0.05:
        raise SystemExit("The approved circles experiment did not satisfy closure criteria.")
    circles_final_boundary = decision_boundary(
        circles_model,
        circles.features,
        resolution=circles_config.grid_resolution,
    )
    render_boundary_svg(
        "artifacts/figures/demo-circles-before.svg",
        title="Circles decision boundary · before training",
        boundary=circles_initial_boundary,
        features=circles.features,
        targets=circles.targets,
    )
    render_boundary_svg(
        "artifacts/figures/demo-circles-after.svg",
        title="Circles decision boundary · after training",
        boundary=circles_final_boundary,
        features=circles.features,
        targets=circles.targets,
    )
    summary = {
        "status": "passed",
        "tests": {
            "gradient_check": gradient_report.to_dict(),
            "xor_accuracy": accuracy,
            "initial_loss": history.loss[0],
            "final_loss": history.loss[-1],
            "checkpoint_reconstruction": True,
            "trace_schema": trace["schema_version"],
            "pytorch_parity": parity.to_dict(),
            "circles_accuracy": circles_accuracy,
            "circles_initial_loss": circles_history.loss[0],
            "circles_final_loss": circles_history.loss[-1],
        },
        "limitations": [
            "XOR contains four pedagogical observations.",
            "Passing parity covers the approved MLP, not every PyTorch feature.",
            "Training accuracy is not a generalization claim.",
        ],
    }
    destination = write_json(
        "artifacts/comparisons/demo-validation-summary.json",
        summary,
    )
    print(json.dumps(summary, indent=2))
    print(f"Validation evidence: {destination.resolve()}")


if __name__ == "__main__":
    main()
