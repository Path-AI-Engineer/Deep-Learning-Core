from __future__ import annotations

import argparse
import json
import sys
import zipfile

import numpy as np

from _common import PROJECT_ROOT, SRC_ROOT, read_json, write_json

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from sequence_models.contracts import CHANNELS, CLASS_MAPPING  # noqa: E402
from sequence_models.data import load_uci_har, prepare_grouped_splits  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare leakage-safe UCI HAR splits.")
    parser.add_argument("--validation-subjects", default="1,3,5,7")
    args = parser.parse_args()
    raw_root = PROJECT_ROOT / "data" / "raw"
    archive = raw_root / "uci_har.zip"
    extraction = raw_root / "extracted"
    if not extraction.exists():
        if not archive.is_file():
            raise FileNotFoundError("Run scripts/download_data.py first.")
        with zipfile.ZipFile(archive) as handle:
            handle.extractall(extraction)
    official_train, official_test = load_uci_har(extraction)
    validation_subjects = tuple(int(value) for value in args.validation_subjects.split(","))
    prepared = prepare_grouped_splits(
        official_train,
        official_test,
        validation_subjects=validation_subjects,
    )
    output = PROJECT_ROOT / "data" / "processed" / "uci_har_prepared.npz"
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        train_values=prepared.train.values,
        train_labels=prepared.train.labels,
        train_subjects=prepared.train.subjects,
        validation_values=prepared.validation.values,
        validation_labels=prepared.validation.labels,
        validation_subjects=prepared.validation.subjects,
        test_values=prepared.test.values,
        test_labels=prepared.test.labels,
        test_subjects=prepared.test.subjects,
        mean=prepared.mean,
        std=prepared.std,
    )
    download_manifest = read_json(raw_root / "download_manifest.json")
    manifest = {
        "dataset": download_manifest,
        "shape": [128, len(CHANNELS)],
        "channels": list(CHANNELS),
        "class_mapping": CLASS_MAPPING,
        "split_strategy": "official train/test; grouped validation subjects from official train",
        "validation_subjects": list(validation_subjects),
        "sizes": {
            "train": len(prepared.train.values),
            "validation": len(prepared.validation.values),
            "test": len(prepared.test.values),
        },
        "normalization": "per-channel z-score fitted only on the training partition",
        "mean": prepared.mean.tolist(),
        "std": prepared.std.tolist(),
    }
    write_json(output.parent / "preparation_manifest.json", manifest)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

