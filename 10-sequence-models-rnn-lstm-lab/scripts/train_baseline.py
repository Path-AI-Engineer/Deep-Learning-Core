from __future__ import annotations

import argparse
import json

from train_model import run_training


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the temporal-statistics MLP baseline.")
    parser.add_argument("--version", default="v1.1.0-uci")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            run_training("statistics-mlp", args.version, args.force),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

