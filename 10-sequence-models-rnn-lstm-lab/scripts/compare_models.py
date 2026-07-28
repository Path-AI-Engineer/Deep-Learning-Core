from __future__ import annotations

import argparse
import json
import sys

from _common import PROJECT_ROOT, SRC_ROOT, write_json

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from sequence_models.artifacts import load_bundle  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare version-aligned model bundles.")
    parser.add_argument("--version", default="v1.1.0-uci")
    args = parser.parse_args()
    rows: list[dict[str, object]] = []
    for model_name in ("statistics-mlp", "rnn", "lstm", "gru"):
        bundle = load_bundle(
            PROJECT_ROOT / "artifacts" / "models" / model_name / args.version
        )
        rows.append({"model_id": model_name, **bundle.metrics})
    approved = max(rows, key=lambda row: float(row["validation_macro_f1"]))
    payload = {
        "comparison_version": args.version,
        "data_mode": "official_uci_har",
        "selection_metric": "validation_macro_f1",
        "approved_model": approved["model_id"],
        "models": rows,
    }
    destination = PROJECT_ROOT / "artifacts" / "comparisons" / args.version
    write_json(destination / "model_comparison.json", payload)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

