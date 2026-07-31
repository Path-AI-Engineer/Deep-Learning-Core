from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from transformer_lab.artifacts import load_bundle


def main() -> None:
    bundle = load_bundle(Path("artifacts/models/transformer/v1.0.0-reference"))
    metrics = bundle.files["metrics.json"]
    per_task = bundle.files["per_task_metrics.json"]
    data_manifest = bundle.files["data_manifest.json"]
    output = Path("paper/data/reference-validation.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {
                "source_bundle": bundle.manifest["model_version"],
                "evidence_status": bundle.manifest["evidence_status"],
                "metrics": metrics,
                "per_task": per_task,
                "data_manifest": data_manifest,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    table = Path("paper/tables/reference-validation.md")
    table.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        "| Split | Exact match | Token accuracy | Count |",
        "|---|---:|---:|---:|",
    ]
    for split in ("validation_id", "validation_ood"):
        values = metrics[split]
        rows.append(
            f"| {split} | {values['exact_match']:.4f} | "
            f"{values['token_accuracy']:.4f} | {values['count']} |"
        )
    table.write_text("\n".join(rows) + "\n", encoding="utf-8")
    print(json.dumps({"status": "completed", "data": str(output), "table": str(table)}))


if __name__ == "__main__":
    main()
