from __future__ import annotations

import argparse
import json
import platform
import shutil
import sys
import time
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from transformer_lab.artifacts import write_bundle
from transformer_lab.contracts import ModelConfig, SequenceExample
from transformer_lab.data import assert_disjoint, generate_balanced_suite
from transformer_lab.decoding import greedy_decode
from transformer_lab.evaluation import evaluate_sequence
from transformer_lab.models import SequenceTransformer
from transformer_lab.tokenization import Vocabulary
from transformer_lab.training import result_metadata, train


def _load_config(path: Path) -> ModelConfig:
    return ModelConfig(**json.loads(path.read_text(encoding="utf-8")))


def _evaluate(
    model: SequenceTransformer,
    examples: list[SequenceExample],
    *,
    max_examples_per_task: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    vocabulary = Vocabulary()
    selected: list[SequenceExample] = []
    task_counts: defaultdict[str, int] = defaultdict(int)
    for example in examples:
        if task_counts[example.task] >= max_examples_per_task:
            continue
        selected.append(example)
        task_counts[example.task] += 1
    rows: list[dict[str, Any]] = []
    for example in selected:
        source = torch.tensor(
            [vocabulary.encode(example.source_tokens)],
            dtype=torch.long,
        )
        result = greedy_decode(
            model,
            source,
            max_new_tokens=min(model.config.max_length - 1, 28),
        )
        target = vocabulary.encode(example.target_tokens[1:])
        metrics = evaluate_sequence(result.token_ids, target)
        rows.append(
            {
                "example_id": example.example_id,
                "task": example.task,
                "split": example.split,
                "length": example.content_length,
                "prediction": vocabulary.decode(result.token_ids),
                "target": list(example.target_tokens[1:]),
                "stopped_by": result.stopped_by,
                **metrics.as_dict(),
            }
        )
    per_task: dict[str, dict[str, float | int]] = {}
    per_length: dict[str, dict[str, float | int]] = {}
    for key_function, destination in (
        (lambda row: row["task"], per_task),
        (lambda row: str(row["length"]), per_length),
    ):
        grouped: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            grouped[str(key_function(row))].append(row)
        for key, values in grouped.items():
            destination[key] = {
                "count": len(values),
                "exact_match": sum(float(row["exact_match"]) for row in values)
                / len(values),
                "token_accuracy": sum(float(row["token_accuracy"]) for row in values)
                / len(values),
            }
    aggregate = {
        "count": len(rows),
        "exact_match": sum(float(row["exact_match"]) for row in rows) / len(rows),
        "token_accuracy": sum(float(row["token_accuracy"]) for row in rows) / len(rows),
        "per_task": per_task,
        "per_length": per_length,
    }
    return aggregate, rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and package the reference Transformer.")
    parser.add_argument("--config", type=Path, default=Path("configs/models/reference.json"))
    parser.add_argument(
        "--destination",
        type=Path,
        default=Path("artifacts/models/transformer/v1.0.0-reference"),
    )
    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--count-per-task", type=int, default=128)
    parser.add_argument("--seed", type=int, default=1200)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.destination.exists() and args.force:
        shutil.rmtree(args.destination)
    config = _load_config(args.config)
    torch.manual_seed(args.seed)
    training = generate_balanced_suite(
        seed=args.seed,
        count_per_task=args.count_per_task,
        split="train",
        copy_reverse_range=(5, 16),
        recall_range=(2, 6),
    )
    validation_id = generate_balanced_suite(
        seed=args.seed + 1,
        count_per_task=max(args.count_per_task // 6, 12),
        split="validation_id",
        copy_reverse_range=(5, 16),
        recall_range=(2, 6),
    )
    validation_ood = generate_balanced_suite(
        seed=args.seed + 2,
        count_per_task=max(args.count_per_task // 10, 8),
        split="validation_ood",
        copy_reverse_range=(17, 20),
        recall_range=(7, 8),
    )
    assert_disjoint(training, validation_id, validation_ood)
    model = SequenceTransformer(config)
    result = train(
        model,
        training,
        validation_id,
        epochs=args.epochs,
        batch_size=32,
        seed=args.seed,
    )
    id_metrics, id_predictions = _evaluate(
        model, validation_id, max_examples_per_task=8
    )
    ood_metrics, ood_predictions = _evaluate(
        model, validation_ood, max_examples_per_task=5
    )
    vocabulary = Vocabulary()
    latency_sample = validation_id[0]
    source = torch.tensor(
        [vocabulary.encode(latency_sample.source_tokens)],
        dtype=torch.long,
    )
    timings: list[float] = []
    for _ in range(8):
        started = time.perf_counter()
        greedy_decode(model, source, max_new_tokens=20)
        timings.append((time.perf_counter() - started) * 1000)
    timings.sort()
    data_manifest = {
        "name": "Controlled Sequence Transduction Suite",
        "version": "1.0.0",
        "status": "validation_only",
        "train_examples": len(training),
        "validation_id_examples": len(validation_id),
        "validation_ood_examples": len(validation_ood),
        "test_status": "frozen_not_opened",
        "train_lengths": {"copy_reverse": [5, 16], "recall_pairs": [2, 6]},
        "validation_ood_lengths": {"copy_reverse": [17, 20], "recall_pairs": [7, 8]},
    }
    documents = {
        "model_config.json": config.as_dict(),
        "vocabulary.json": vocabulary.as_dict(),
        "task_config.json": {
            "tasks": ["copy", "reverse", "recall"],
            "oracles": "project-local deterministic functions",
        },
        "preprocessing.json": {
            "tokenization": "fixed discrete vocabulary",
            "batch_first": True,
            "padding_id": 0,
            "target_shift": "BOS input; EOS target; PAD ignored",
        },
        "decoding_config.json": {
            "strategy": "greedy",
            "bos_id": 1,
            "eos_id": 2,
            "max_new_tokens": 28,
        },
        "data_manifest.json": data_manifest,
        "split_manifest.json": {
            "seed": args.seed,
            "disjoint_hashes": True,
            "selection_uses": ["validation_id", "validation_ood"],
            "test_accessed": False,
        },
        "metrics.json": {
            "status": "reference_validation",
            "validation_id": {
                key: value for key, value in id_metrics.items() if not key.startswith("per_")
            },
            "validation_ood": {
                key: value for key, value in ood_metrics.items() if not key.startswith("per_")
            },
            "generalization_gap_exact_match": (
                float(id_metrics["exact_match"]) - float(ood_metrics["exact_match"])
            ),
        },
        "per_task_metrics.json": {
            "validation_id": id_metrics["per_task"],
            "validation_ood": ood_metrics["per_task"],
        },
        "per_length_metrics.json": {
            "validation_id": id_metrics["per_length"],
            "validation_ood": ood_metrics["per_length"],
        },
        "latency.json": {
            "device": "cpu",
            "batch_size": 1,
            "samples": len(timings),
            "median_ms": timings[len(timings) // 2],
            "p90_ms": timings[-1],
        },
        "training_history.json": result_metadata(result),
    }
    metadata = {
        "model_family": "transformer",
        "model_version": "v1.0.0",
        "evidence_status": "reference_validation",
        "created_at": datetime.now(UTC).isoformat(),
        "seed": args.seed,
        "framework": {"python": platform.python_version(), "torch": torch.__version__},
        "limitations": [
            "Synthetic algorithmic tasks are not natural-language understanding.",
            "Validation results are not final frozen-test results.",
            "Attention weights are descriptive and not causal explanations.",
        ],
    }
    write_bundle(args.destination, model, documents=documents, metadata=metadata)
    sample_rows = validation_id[:12] + validation_ood[:6]
    sample_path = Path("data/samples/demo_catalog.json")
    sample_path.parent.mkdir(parents=True, exist_ok=True)
    sample_path.write_text(
        json.dumps(
            {
                "version": "1.0.0",
                "samples": [example.as_dict() for example in sample_rows],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    prediction_path = Path("reports/predictions/reference-validation.json")
    prediction_path.parent.mkdir(parents=True, exist_ok=True)
    prediction_path.write_text(
        json.dumps(
            {
                "validation_id": id_predictions,
                "validation_ood": ood_predictions,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": "completed",
                "bundle": str(args.destination),
                "validation_id": id_metrics,
                "validation_ood": ood_metrics,
                "test_accessed": False,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
