from __future__ import annotations

from fastapi.testclient import TestClient

from backend.app.main import create_app


def test_health_reports_degraded_when_bundle_is_absent(monkeypatch) -> None:
    monkeypatch.setenv("TRANSFORMER_BUNDLE_PATH", "missing/bundle")
    monkeypatch.setenv("TRANSFORMER_SAMPLE_PATH", "missing/samples.json")
    with TestClient(create_app()) as client:
        response = client.get("/api/v1/health")
    assert response.status_code == 200
    assert response.json()["status"] == "degraded"


def test_attention_fixture_returns_normalized_weights() -> None:
    with TestClient(create_app()) as client:
        response = client.post(
            "/api/v1/attention/compute",
            json={
                "query": [[1.0, 0.0], [0.0, 1.0]],
                "key": [[1.0, 0.0], [0.0, 1.0]],
                "value": [[1.0, 2.0], [3.0, 4.0]],
                "mask": [[False, True], [False, False]],
            },
        )
    assert response.status_code == 200
    payload = response.json()
    assert payload["weights"][0] == [1.0, 0.0]
    assert payload["reference_difference"] < 1e-12


def test_custom_input_validation_is_bounded() -> None:
    with TestClient(create_app()) as client:
        response = client.post(
            "/api/v1/predict",
            json={"task": "copy", "source_symbols": ["UNKNOWN"], "max_new_tokens": 8},
        )
    assert response.status_code in {422, 503}
