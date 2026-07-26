from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pytorch_tabular.data import load_task_data


def main() -> None:
    destination = ROOT / "data" / "processed"
    destination.mkdir(parents=True, exist_ok=True)
    summary: dict[str, object] = {}
    for task in ("regression", "classification"):
        prepared = load_task_data(task)  # type: ignore[arg-type]
        summary[task] = {
            "dataset": prepared.dataset_name,
            "features": prepared.feature_names,
            "splits": prepared.split_summary(),
        }
    path = destination / "dataset-summary.json"
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(path.resolve())


if __name__ == "__main__":
    main()
