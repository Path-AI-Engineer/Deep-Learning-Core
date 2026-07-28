from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health_exposes_bundles_without_paths() -> None:
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ready"
    assert set(payload["bundles_available"]) == {"rnn", "lstm", "gru"}
    assert "C:\\" not in response.text


def test_samples_and_detail_keep_sequence_contract() -> None:
    samples = client.get("/api/v1/samples?limit=3").json()["items"]
    assert len(samples) == 3
    detail = client.get(f"/api/v1/samples/{samples[0]['sample_id']}")
    assert detail.status_code == 200
    assert len(detail.json()["signals"]) == 9
    assert len(detail.json()["signals"][0]) == 128


def test_prediction_and_comparison_use_whitelisted_models() -> None:
    payload = {"sample_id": "demo-0-00", "model_id": "rnn"}
    prediction = client.post("/api/v1/predictions/sample", json=payload)
    assert prediction.status_code == 200
    assert len(prediction.json()["probabilities"]) == 6
    comparison = client.post(
        "/api/v1/predictions/compare",
        json={"sample_id": "demo-0-00", "model_ids": ["rnn", "lstm", "gru"]},
    )
    assert comparison.status_code == 200
    assert len(comparison.json()["predictions"]) == 3


def test_cell_and_gradient_labs_are_bounded() -> None:
    trace = client.post(
        "/api/v1/labs/cell-trace",
        json={"cell_type": "lstm", "example_id": "balanced-memory"},
    )
    assert trace.status_code == 200
    assert trace.json()["max_abs_difference"] <= trace.json()["parity_tolerance"]
    gradient = client.get("/api/v1/labs/gradient-flow")
    assert gradient.status_code == 200
    assert gradient.json()["finite"] is True


def test_invalid_ids_and_oversized_trace_are_rejected() -> None:
    assert client.get("/api/v1/samples/not-real").status_code == 404
    invalid_model = client.post(
        "/api/v1/predictions/sample",
        json={"sample_id": "demo-0-00", "model_id": "transformer"},
    )
    assert invalid_model.status_code == 422
    oversized = client.post(
        "/api/v1/traces/sample",
        json={
            "sample_id": "demo-0-00",
            "model_id": "rnn",
            "selected_units": [0],
            "start_timestep": 0,
            "end_timestep": 100,
        },
    )
    assert oversized.status_code == 422


def test_direct_spa_route_has_fallback_after_build() -> None:
    response = client.get("/sequence-lab", headers={"accept": "text/html"})
    assert response.status_code == 200
    assert "Sequence Memory Lab" in response.text
