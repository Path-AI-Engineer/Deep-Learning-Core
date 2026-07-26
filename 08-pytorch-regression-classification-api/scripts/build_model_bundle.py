from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pytorch_tabular.artifacts import validate_bundle


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate an approved model bundle.")
    parser.add_argument("bundle", type=Path)
    args = parser.parse_args()
    metadata = validate_bundle(args.bundle)
    print(f"Bundle valid: {args.bundle.resolve()}")
    print(f"Task: {metadata['task']} | Model: {metadata['model_version']}")


if __name__ == "__main__":
    main()
