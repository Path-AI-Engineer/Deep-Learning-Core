from __future__ import annotations

import argparse
import json
from pathlib import Path

from transformer_lab.data import assert_disjoint, generate_balanced_suite


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate controlled sequence splits.")
    parser.add_argument("--output", type=Path, default=Path("data/manifests/suite.json"))
    parser.add_argument("--count-per-task", type=int, default=128)
    parser.add_argument("--seed", type=int, default=1200)
    args = parser.parse_args()
    training = generate_balanced_suite(
        seed=args.seed,
        count_per_task=args.count_per_task,
        split="train",
        copy_reverse_range=(5, 16),
        recall_range=(2, 6),
    )
    validation_id = generate_balanced_suite(
        seed=args.seed + 1,
        count_per_task=max(args.count_per_task // 6, 6),
        split="validation_id",
        copy_reverse_range=(5, 16),
        recall_range=(2, 6),
    )
    validation_ood = generate_balanced_suite(
        seed=args.seed + 2,
        count_per_task=max(args.count_per_task // 8, 4),
        split="validation_ood",
        copy_reverse_range=(17, 20),
        recall_range=(7, 8),
    )
    assert_disjoint(training, validation_id, validation_ood)
    payload = {
        "name": "Controlled Sequence Transduction Suite",
        "version": "1.0.0",
        "seed": args.seed,
        "status": "validation_protocol",
        "splits": {
            "train": [example.as_dict() for example in training],
            "validation_id": [example.as_dict() for example in validation_id],
            "validation_ood": [example.as_dict() for example in validation_ood],
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": "ready",
                "output": str(args.output),
                "counts": {
                    split: len(rows)
                    for split, rows in payload["splits"].items()
                },
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

