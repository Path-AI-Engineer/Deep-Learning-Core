from __future__ import annotations

import io

from fastapi.testclient import TestClient
from PIL import Image


def test_health_contract(client: TestClient) -> None:
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ready"
    assert body["data_mode"] == "educational_fixture"
    assert body["active_model"] == "conv-ae"


def test_model_card_declares_limitations(client: TestClient) -> None:
    body = client.get("/api/v1/model-card").json()
    assert body["labels_used_for_autoencoder_training"] is False
    assert len(body["limitations"]) >= 4


def test_sample_gallery_and_detail(client: TestClient) -> None:
    gallery = client.get("/api/v1/samples", params={"class_id": 8, "limit": 2})
    assert gallery.status_code == 200
    items = gallery.json()["items"]
    assert len(items) == 2
    detail = client.get(f"/api/v1/samples/{items[0]['sample_id']}")
    assert detail.status_code == 200
    assert detail.json()["label"] == 8


def test_reconstruct_sample(client: TestClient) -> None:
    response = client.post(
        "/api/v1/reconstruct/sample",
        json={"sample_id": "fixture-0-13", "model_id": "conv-ae"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["model_id"] == "conv-ae"
    assert body["metrics"]["mse"] >= 0


def test_ephemeral_upload_contract(client: TestClient) -> None:
    buffer = io.BytesIO()
    Image.new("L", (28, 28), color=128).save(buffer, format="PNG")
    response = client.post(
        "/api/v1/reconstruct/upload",
        params={"model_id": "conv-ae"},
        content=buffer.getvalue(),
        headers={"Content-Type": "image/png"},
    )
    assert response.status_code == 200
    assert response.json()["persisted"] is False


def test_denoise_contract(client: TestClient) -> None:
    response = client.post(
        "/api/v1/denoise",
        json={
            "sample_id": "fixture-0-13",
            "corruption_type": "gaussian",
            "corruption_level": 0.2,
            "seed": 42,
            "model_ids": ["conv-ae", "denoising-ae"],
        },
    )
    assert response.status_code == 200
    assert len(response.json()["reconstructions"]) == 2


def test_latent_points_and_sample(client: TestClient) -> None:
    points = client.get("/api/v1/latent/points", params={"limit": 3}).json()["items"]
    assert len(points) == 3
    detail = client.get(f"/api/v1/latent/sample/{points[0]['sample_id']}")
    assert detail.status_code == 200
    assert detail.json()["sample_id"] == points[0]["sample_id"]


def test_interpolation_contract(client: TestClient) -> None:
    response = client.post(
        "/api/v1/latent/interpolate",
        json={
            "model_id": "conv-ae",
            "sample_id_a": "fixture-0-13",
            "sample_id_b": "fixture-8-13",
            "steps": 5,
        },
    )
    assert response.status_code == 200
    assert [row["alpha"] for row in response.json()["items"]] == [0, 0.25, 0.5, 0.75, 1]


def test_evaluation_contracts(client: TestClient) -> None:
    summary = client.get("/api/v1/evaluation/summary")
    model = client.get("/api/v1/evaluation/model/conv-ae")
    errors = client.get(
        "/api/v1/evaluation/errors",
        params={"model_id": "conv-ae", "limit": 4},
    )
    assert summary.status_code == model.status_code == errors.status_code == 200
    assert len(summary.json()["models"]) == 6
    assert len(errors.json()["items"]) == 4


def test_validation_error_is_bounded(client: TestClient) -> None:
    response = client.post(
        "/api/v1/reconstruct/sample",
        json={"sample_id": "missing", "model_id": "conv-ae"},
    )
    assert response.status_code == 422
    assert response.json()["code"] == "validation_error"
