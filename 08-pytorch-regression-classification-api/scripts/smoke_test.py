from __future__ import annotations

import argparse
import json
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def request_json(
    base_url: str,
    path: str,
    *,
    method: str = "GET",
    payload: dict[str, object] | None = None,
) -> dict[str, object]:
    body = json.dumps(payload).encode() if payload is not None else None
    request = Request(
        f"{base_url.rstrip('/')}{path}",
        data=body,
        method=method,
        headers={"Content-Type": "application/json"},
    )
    with urlopen(request, timeout=15) as response:
        return json.loads(response.read())


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke-test a running Project 08 app.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8008")
    args = parser.parse_args()
    try:
        health = request_json(args.base_url, "/api/v1/health")
        assert health["status"] == "ready"
        for task in ("regression", "classification"):
            schema = request_json(args.base_url, f"/api/v1/tasks/{task}/schema")
            example = schema["examples"][0]
            prediction = request_json(
                args.base_url,
                f"/api/v1/predictions/{task}",
                method="POST",
                payload={"features": example},
            )
            assert prediction["model_version"] == "1.0.0"
        batch_schema = request_json(
            args.base_url, "/api/v1/tasks/classification/schema"
        )
        batch = request_json(
            args.base_url,
            "/api/v1/predictions/classification/batch",
            method="POST",
            payload={"rows": batch_schema["examples"] * 2},
        )
        assert batch["count"] == 2
    except (AssertionError, HTTPError, URLError, KeyError, TypeError) as error:
        raise SystemExit(f"Smoke test failed: {error}") from error
    print("Smoke passed: health, regression, classification and batch inference.")


if __name__ == "__main__":
    main()
