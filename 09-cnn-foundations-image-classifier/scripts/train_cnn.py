from __future__ import annotations

import argparse
import json
from pathlib import Path

from cnn_foundations.experiments.runner import ExperimentRunner


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate the FashionMNIST CNN.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/experiments/cnn_base.yaml"),
    )
    args = parser.parse_args()
    destination = ExperimentRunner().run(args.config, evaluate_test=True)
    print(json.dumps({"status": "completed", "run_directory": str(destination)}))


if __name__ == "__main__":
    main()
