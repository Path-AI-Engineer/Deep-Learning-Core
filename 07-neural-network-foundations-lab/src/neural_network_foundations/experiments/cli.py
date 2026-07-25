"""CLI entrypoint for reproducible YAML experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml  # type: ignore[import-untyped]

from neural_network_foundations.contracts import ExperimentConfig
from neural_network_foundations.experiments.runner import run_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--artifact-root", type=Path, default=Path("artifacts/runs"))
    parser.add_argument("--run-id")
    parser.add_argument("--without-parity", action="store_true")
    args = parser.parse_args()
    payload = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Experiment configuration must be a YAML object.")
    result = run_experiment(
        ExperimentConfig.from_dict(payload),
        artifact_root=args.artifact_root,
        run_id=args.run_id,
        include_parity=not args.without_parity,
    )
    print(json.dumps(result.to_dict(), indent=2))


if __name__ == "__main__":
    main()
