from __future__ import annotations

import argparse

import numpy as np
import torch

from _common import PROJECT_ROOT, configure_imports

configure_imports()

from autoencoder_lab.artifacts import load_bundle  # noqa: E402
from autoencoder_lab.representations import extract_embeddings  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="conv-ae")
    args = parser.parse_args()
    data_path = PROJECT_ROOT / "data/processed/fashion-mnist/dataset.npz"
    bundle_path = PROJECT_ROOT / "artifacts/official" / args.model / "v1.0.0"
    if not data_path.is_file() or not bundle_path.is_dir():
        raise FileNotFoundError("Prepare data and train the selected model first.")
    bundle = load_bundle(bundle_path)
    with np.load(data_path) as values:
        output = {
            f"{split}_embeddings": extract_embeddings(
                bundle.model,
                torch.from_numpy(values[f"{split}_images"]),
            )
            for split in ("train", "validation", "test")
        }
        output.update(
            {
                f"{split}_labels": values[f"{split}_labels"]
                for split in ("train", "validation", "test")
            }
        )
    destination = PROJECT_ROOT / f"artifacts/official/{args.model}/v1.0.0/embeddings.npz"
    np.savez_compressed(destination, **output)
    print(f"Embeddings written to {destination}")


if __name__ == "__main__":
    main()
