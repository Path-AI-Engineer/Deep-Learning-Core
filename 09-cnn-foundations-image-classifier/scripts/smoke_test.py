from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "backend")]

from fastapi.testclient import TestClient  # noqa: E402

from app.main import create_app  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke-test the packaged API.")
    parser.add_argument(
        "--bundle", type=Path, default=Path("artifacts/models/cnn/v1.0.0")
    )
    parser.add_argument("--data-root", type=Path, default=Path("data/raw"))
    args = parser.parse_args()
    with TestClient(create_app(bundle_path=args.bundle, data_root=args.data_root)) as client:
        health = client.get("/api/v1/health")
        health.raise_for_status()
        payload = health.json()
        if payload["status"] != "ready":
            raise RuntimeError(f"application is not ready: {payload}")
        classes = client.get("/api/v1/classes")
        classes.raise_for_status()
        prediction = client.post(
            "/api/v1/predictions/sample", json={"sample_id": "test-00000"}
        )
        prediction.raise_for_status()
        print(
            json.dumps(
                {
                    "status": "passed",
                    "model_version": payload["model_version"],
                    "predicted_class": prediction.json()["predicted_class"],
                }
            )
        )


if __name__ == "__main__":
    main()
