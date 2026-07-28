from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
import zipfile
from datetime import UTC, datetime
from pathlib import Path

from _common import PROJECT_ROOT, write_json

DATASET_URL = (
    "https://archive.ics.uci.edu/static/public/240/"
    "human+activity+recognition+using+smartphones.zip"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Download the official UCI HAR archive.")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    destination = PROJECT_ROOT / "data" / "raw" / "uci_har.zip"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not args.force:
        raise FileExistsError(f"{destination} exists; pass --force to download it again")
    request = urllib.request.Request(DATASET_URL, headers={"User-Agent": "SequenceMemoryLab/1.0"})
    with urllib.request.urlopen(request, timeout=120) as response:
        destination.write_bytes(response.read())
    if not zipfile.is_zipfile(destination):
        destination.unlink(missing_ok=True)
        raise ValueError("Downloaded content is not a valid ZIP archive")
    manifest = {
        "dataset": "UCI Human Activity Recognition Using Smartphones",
        "source": DATASET_URL,
        "downloaded_at": datetime.now(UTC).isoformat(),
        "archive": destination.name,
        "bytes": destination.stat().st_size,
        "sha256": sha256(destination),
    }
    write_json(destination.parent / "download_manifest.json", manifest)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

