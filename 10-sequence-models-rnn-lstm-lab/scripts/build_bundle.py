from __future__ import annotations

import argparse
import json
import sys

from _common import PROJECT_ROOT, SRC_ROOT

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from sequence_models.artifacts import load_bundle  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a completed immutable bundle.")
    parser.add_argument("--model", choices=("statistics-mlp", "rnn", "lstm", "gru"), required=True)
    parser.add_argument("--version", default="v1.1.0-uci")
    args = parser.parse_args()
    directory = PROJECT_ROOT / "artifacts" / "models" / args.model / args.version
    bundle = load_bundle(directory)
    print(
        json.dumps(
            {
                "bundle": str(directory),
                "model_id": bundle.model_id,
                "version": bundle.version,
                "state_sha256": bundle.manifest["state_sha256"],
                "status": "validated",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

