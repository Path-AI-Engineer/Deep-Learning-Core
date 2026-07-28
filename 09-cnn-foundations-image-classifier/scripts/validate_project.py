from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "backend")]

from app.main import create_app  # noqa: E402
from cnn_foundations.contracts.config import load_experiment_config  # noqa: E402


def main() -> None:
    for config in (
        ROOT / "configs/experiments/mlp_baseline.yaml",
        ROOT / "configs/experiments/cnn_base.yaml",
    ):
        load_experiment_config(config.relative_to(ROOT))
    schema = create_app().openapi()
    required = {
        "/api/v1/health",
        "/api/v1/model-card",
        "/api/v1/classes",
        "/api/v1/samples",
        "/api/v1/predictions/sample",
        "/api/v1/predictions/upload",
        "/api/v1/labs/convolution",
        "/api/v1/explanations/activations",
        "/api/v1/evaluation/summary",
        "/api/v1/evaluation/errors",
    }
    missing = required - set(schema["paths"])
    if missing:
        raise RuntimeError(f"OpenAPI routes missing: {sorted(missing)}")
    print(json.dumps({"status": "passed", "routes": len(schema["paths"])}))


if __name__ == "__main__":
    main()
