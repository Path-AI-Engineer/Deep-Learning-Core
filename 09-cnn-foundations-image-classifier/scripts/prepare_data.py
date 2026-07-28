from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from torchvision.datasets import FashionMNIST


def digest_files(root: Path) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        hashes[path.relative_to(root).as_posix()] = digest
    return hashes


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and verify FashionMNIST.")
    parser.add_argument("--root", type=Path, default=Path("data/raw"))
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/processed/fashion_mnist_manifest.json"),
    )
    args = parser.parse_args()
    train = FashionMNIST(root=args.root, train=True, download=True)
    test = FashionMNIST(root=args.root, train=False, download=True)
    payload = {
        "dataset": "FashionMNIST",
        "source": "torchvision.datasets.FashionMNIST",
        "train_examples": len(train),
        "test_examples": len(test),
        "class_names": list(train.classes),
        "sha256": digest_files(args.root),
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": "ready", **payload}, sort_keys=True))


if __name__ == "__main__":
    main()
