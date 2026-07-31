from __future__ import annotations

import json

from _common import PROJECT_ROOT


def main() -> None:
    root = PROJECT_ROOT / "artifacts/official"
    rows: list[dict[str, object]] = []
    for metrics_path in sorted(root.glob("*/v1.0.0/metrics.json")):
        value = json.loads(metrics_path.read_text(encoding="utf-8"))
        rows.append({"model_id": metrics_path.parents[1].name, "metrics": value})
    if not rows:
        raise FileNotFoundError("No official model metrics are available.")
    output = root / "model_comparison.json"
    output.write_text(json.dumps({"models": rows}, indent=2), encoding="utf-8")
    print(json.dumps({"models_evaluated": len(rows), "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
