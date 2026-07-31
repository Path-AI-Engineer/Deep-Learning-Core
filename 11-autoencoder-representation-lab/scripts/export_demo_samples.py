from __future__ import annotations

import json

import numpy as np

from _common import PROJECT_ROOT


def main() -> None:
    dataset = PROJECT_ROOT / "data/processed/fashion-mnist/dataset.npz"
    if not dataset.is_file():
        raise FileNotFoundError("Run scripts/prepare_data.py first.")
    with np.load(dataset) as values:
        images = values["test_images"][:50]
        labels = values["test_labels"][:50]
    destination = PROJECT_ROOT / "data/samples/official_demo.npz"
    np.savez_compressed(destination, images=images, labels=labels)
    manifest = {
        "source": "official FashionMNIST test partition",
        "count": len(images),
        "selection": "first 50 test records; labels retained for display only",
    }
    destination.with_suffix(".json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    print(f"Demo samples written to {destination}")


if __name__ == "__main__":
    main()
