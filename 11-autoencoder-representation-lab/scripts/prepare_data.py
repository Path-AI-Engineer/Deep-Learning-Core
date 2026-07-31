from __future__ import annotations

import json

import numpy as np

from _common import PROJECT_ROOT, configure_imports

configure_imports()

from autoencoder_lab.data.fashion_mnist import (  # noqa: E402
    checksum_files,
    prepare_fashion_mnist,
    write_split_manifest,
)


def main() -> None:
    raw = PROJECT_ROOT / "data" / "raw"
    output = PROJECT_ROOT / "data" / "processed" / "fashion-mnist"
    output.mkdir(parents=True, exist_ok=True)
    prepared = prepare_fashion_mnist(raw, download=False)
    np.savez_compressed(
        output / "dataset.npz",
        train_images=prepared.train_images,
        train_labels=prepared.train_labels,
        validation_images=prepared.validation_images,
        validation_labels=prepared.validation_labels,
        test_images=prepared.test_images,
        test_labels=prepared.test_labels,
    )
    write_split_manifest(output / "split_manifest.json", prepared)
    (output / "raw_checksums.json").write_text(
        json.dumps(checksum_files(raw), indent=2),
        encoding="utf-8",
    )
    print(f"Prepared FashionMNIST at {output}")


if __name__ == "__main__":
    main()
