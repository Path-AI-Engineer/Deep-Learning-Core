from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pytorch_tabular.artifacts import build_bundle
from pytorch_tabular.contracts import ExperimentConfig
from pytorch_tabular.experiments import run_experiment


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train and package the classification MLP."
    )
    parser.add_argument(
        "--replace",
        action="store_true",
        help="Replace the existing v1.0.0 bundle after a new accepted run.",
    )
    args = parser.parse_args()
    result = run_experiment(
        ExperimentConfig(
            task="classification",
            batch_size=24,
            epochs=160,
            patience=18,
        ),
        ROOT,
    )
    if result.metrics["macro_f1"] <= result.baseline_metrics["macro_f1"]:
        raise SystemExit("Classifier did not beat the prior baseline on macro-F1.")
    bundle = build_bundle(
        result,
        ROOT / "artifacts" / "models" / "classification" / "v1.0.0",
        overwrite=args.replace,
    )
    print(json.dumps({"run_id": result.run_id, "metrics": result.metrics}, indent=2))
    print(f"Bundle: {bundle}")


if __name__ == "__main__":
    main()
