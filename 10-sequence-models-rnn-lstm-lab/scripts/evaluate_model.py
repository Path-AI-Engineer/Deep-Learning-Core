from __future__ import annotations

import argparse
import json
import sys

from _common import PROJECT_ROOT, SRC_ROOT, load_prepared, loader

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from sequence_models.artifacts import load_bundle  # noqa: E402
from sequence_models.evaluation import classification_metrics  # noqa: E402
from sequence_models.training import evaluate  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Re-evaluate an immutable model bundle.")
    parser.add_argument("--model", choices=("statistics-mlp", "rnn", "lstm", "gru"), required=True)
    parser.add_argument("--version", default="v1.1.0-uci")
    args = parser.parse_args()
    prepared = load_prepared()
    bundle = load_bundle(
        PROJECT_ROOT / "artifacts" / "models" / args.model / args.version
    )
    test_loader = loader(prepared["test_values"], prepared["test_labels"], 128, False)
    test_loss, truth, prediction = evaluate(bundle.model, test_loader)
    payload = classification_metrics(truth, prediction).to_dict()
    payload["test_loss"] = round(test_loss, 6)
    payload["bundle"] = f"{args.model}/{args.version}"
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

