"""Generate a NumPy/PyTorch parity report for the approved XOR configuration."""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from neural_network_foundations.contracts import ExperimentConfig
from neural_network_foundations.datasets import get_dataset
from neural_network_foundations.models import MLP
from neural_network_foundations.models.pytorch_reference import compare_with_pytorch
from neural_network_foundations.serialization import write_json


def main() -> None:
    config = ExperimentConfig()
    dataset = get_dataset(config.dataset)
    model = MLP(config.network, seed=config.seed)
    report = compare_with_pytorch(
        model,
        dataset.features,
        dataset.targets,
        loss_name=config.loss,
        learning_rate=config.learning_rate,
    )
    path = write_json("artifacts/comparisons/demo-pytorch-parity.json", report.to_dict())
    print(f"PyTorch parity: {'passed' if report.passed else 'failed'}")
    print(f"Report: {path.resolve()}")
    if not report.passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
