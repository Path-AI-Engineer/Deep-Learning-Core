from __future__ import annotations

import argparse
import json

import numpy as np

from _common import PROJECT_ROOT, configure_imports

configure_imports()

from autoencoder_lab.representations import train_linear_probe  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="conv-ae")
    args = parser.parse_args()
    path = PROJECT_ROOT / f"artifacts/official/{args.model}/v1.0.0/embeddings.npz"
    if not path.is_file():
        raise FileNotFoundError("Run scripts/extract_embeddings.py first.")
    with np.load(path) as values:
        metrics = train_linear_probe(
            values["train_embeddings"],
            values["train_labels"],
            values["test_embeddings"],
            values["test_labels"],
        )
    output = path.parent / "linear_probe_metrics.json"
    output.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
