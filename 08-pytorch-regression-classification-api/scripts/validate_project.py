from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "backend")]

from app.main import create_app
from pytorch_tabular.artifacts import validate_bundle
from pytorch_tabular.inference import ModelPredictor


def main() -> None:
    failures: list[str] = []
    for task in ("regression", "classification"):
        bundle = ROOT / "artifacts" / "models" / task / "v1.0.0"
        try:
            metadata = validate_bundle(bundle)
            predictor = ModelPredictor(bundle)
            predictor.predict([metadata["examples"][0]])
        except (KeyError, OSError, RuntimeError, TypeError, ValueError) as error:
            failures.append(f"{task} bundle: {error}")

    schema = create_app().openapi()
    required_paths = {
        "/api/v1/health",
        "/api/v1/tasks",
        "/api/v1/tasks/{task}/schema",
        "/api/v1/tasks/{task}/model-card",
        "/api/v1/predictions/{task}",
        "/api/v1/predictions/{task}/batch",
    }
    missing_paths = required_paths - set(schema["paths"])
    if missing_paths:
        failures.append(f"OpenAPI paths missing: {sorted(missing_paths)}")
    if not (ROOT / "frontend" / "dist" / "index.html").is_file():
        failures.append("Frontend production build is missing.")

    if failures:
        raise SystemExit("\n".join(failures))
    print("Project contract validation passed.")
    print("Bundles: regression and classification")
    print("OpenAPI: required v1 routes present")
    print("Frontend: production build present")


if __name__ == "__main__":
    main()
