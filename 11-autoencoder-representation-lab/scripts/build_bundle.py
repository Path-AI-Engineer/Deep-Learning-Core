from __future__ import annotations

import argparse
import json

from _common import PROJECT_ROOT, configure_imports

configure_imports()

from autoencoder_lab.artifacts import load_bundle  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="conv-ae")
    parser.add_argument("--source", choices=["models", "official"], default="models")
    args = parser.parse_args()
    directory = PROJECT_ROOT / "artifacts" / args.source / args.model / "v1.0.0"
    bundle = load_bundle(directory)
    print(
        json.dumps(
            {
                "status": "valid",
                "model_id": bundle.model_id,
                "version": bundle.version,
                "state_sha256": bundle.manifest["state_sha256"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
