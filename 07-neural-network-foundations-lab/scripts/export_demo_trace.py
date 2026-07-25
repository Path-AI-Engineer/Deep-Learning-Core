"""Export a real forward trace for the Neural Network Explainer."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from neural_network_foundations.contracts import ExperimentConfig
from neural_network_foundations.datasets import get_dataset
from neural_network_foundations.models import MLP
from neural_network_foundations.serialization import write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["and", "or", "xor", "circles"], default="xor")
    parser.add_argument("--sample-index", type=int, default=1)
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/traces/demo-forward-trace.json")
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = ExperimentConfig(dataset=args.dataset)
    dataset = get_dataset(config.dataset, seed=config.seed)
    model = MLP(config.network, seed=config.seed)
    model.forward(dataset.features)
    model.backward(dataset.targets, loss_name=config.loss)
    trace = model.trace_sample(
        dataset=dataset.name,
        features=dataset.features,
        targets=dataset.targets,
        sample_index=args.sample_index,
        loss_name=config.loss,
        configuration=config.to_dict(),
    )
    destination = write_json(args.output, trace.to_dict())
    print(f"Trace exported: {destination.resolve()}")


if __name__ == "__main__":
    main()
