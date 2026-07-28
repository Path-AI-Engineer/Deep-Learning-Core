from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import torch

from cnn_foundations.artifacts.bundle import write_bundle
from cnn_foundations.models.cnn import model_from_config


def read_run(path: Path) -> dict[str, Any]:
    payload = json.loads((path / "run.json").read_text(encoding="utf-8"))
    if not payload.get("test_evaluated"):
        raise ValueError(f"{path} has no isolated test evaluation.")
    return cast(dict[str, Any], payload)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create the immutable CPU CNN bundle.")
    parser.add_argument("--cnn-run", type=Path, required=True)
    parser.add_argument("--mlp-run", type=Path, required=True)
    parser.add_argument("--version", default="v1.0.0")
    parser.add_argument("--output-root", type=Path, default=Path("artifacts/models/cnn"))
    args = parser.parse_args()
    cnn = read_run(args.cnn_run)
    mlp = read_run(args.mlp_run)
    if cnn["split"] != mlp["split"]:
        raise ValueError("CNN and MLP runs must use the identical split manifest.")
    if cnn["dataset"] != mlp["dataset"]:
        raise ValueError("CNN and MLP runs must use the identical dataset contract.")
    model = model_from_config(cnn["model"])
    state = torch.load(args.cnn_run / "best_state.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    report = cnn["test"]
    destination = args.output_root / args.version
    documents = {
        "model_config.json": cnn["model"],
        "preprocessing.json": {
            "color_mode": "grayscale",
            "resize": [28, 28],
            "mean": cnn["dataset"]["mean"],
            "std": cnn["dataset"]["std"],
            "tensor_shape": [1, 1, 28, 28],
        },
        "class_mapping.json": {
            str(index): name for index, name in enumerate(cnn["dataset"]["class_names"])
        },
        "metrics.json": {
            "accuracy": report["accuracy"],
            "macro_f1": report["macro_f1"],
            "test_examples": sum(row["support"] for row in report["per_class"]),
        },
        "per_class_metrics.json": {"classes": report["per_class"]},
        "confusion_matrix.json": {
            "labels": cnn["dataset"]["class_names"],
            "matrix": report["confusion_matrix"],
        },
        "training_history.json": {"epochs": cnn["history"], "best_epoch": cnn["best_epoch"]},
        "comparison_with_mlp.json": {
            "protocol": "identical dataset, split, optimizer family and isolated test policy",
            "cnn": {"accuracy": report["accuracy"], "macro_f1": report["macro_f1"]},
            "mlp": {
                "accuracy": mlp["test"]["accuracy"],
                "macro_f1": mlp["test"]["macro_f1"],
            },
        },
        "error_analysis.json": cnn["error_analysis"],
        "split_manifest.json": cnn["split"],
    }
    metadata = {
        "model_version": args.version,
        "created_at": datetime.now(UTC).isoformat(),
        "dataset": "FashionMNIST",
        "run_id": cnn["run_id"],
        "baseline_run_id": mlp["run_id"],
        "commit": cnn.get("commit"),
        "limitations": [
            "Valid only for FashionMNIST-like 28x28 grayscale catalog images.",
            "Predictions and activations are descriptive model outputs, not causal explanations.",
            "Uploaded photographs are out of distribution and may receive overconfident scores.",
        ],
    }
    write_bundle(destination, model, metadata=metadata, documents=documents)
    print(json.dumps({"status": "completed", "bundle": str(destination)}))


if __name__ == "__main__":
    main()
