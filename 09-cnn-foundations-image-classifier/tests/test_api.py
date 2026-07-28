from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from app.main import create_app
from app.services.image_validation import extract_image

MISSING = Path("tests/fixtures/assets-that-do-not-exist")


def test_health_is_honestly_degraded_without_assets() -> None:
    with TestClient(
        create_app(bundle_path=MISSING / "bundle", data_root=MISSING / "data")
    ) as client:
        response = client.get("/api/v1/health")
    assert response.status_code == 200
    assert response.json()["status"] == "degraded"
    assert response.headers["X-Request-ID"]


def test_classes_remain_available_without_model() -> None:
    with TestClient(
        create_app(bundle_path=MISSING / "bundle", data_root=MISSING / "data")
    ) as client:
        response = client.get("/api/v1/classes")
    assert response.status_code == 200
    assert len(response.json()["classes"]) == 10


def test_model_endpoint_returns_service_unavailable() -> None:
    with TestClient(
        create_app(bundle_path=MISSING / "bundle", data_root=MISSING / "data")
    ) as client:
        response = client.get("/api/v1/model-card")
    assert response.status_code == 503


def test_convolution_lab_is_independent_and_verified() -> None:
    payload = {
        "matrix": [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
        "kernel": [[1, 0, -1], [1, 0, -1], [1, 0, -1]],
        "stride": 1,
        "padding": 1,
    }
    with TestClient(
        create_app(bundle_path=MISSING / "bundle", data_root=MISSING / "data")
    ) as client:
        response = client.post("/api/v1/labs/convolution", json=payload)
    assert response.status_code == 200
    assert response.json()["parity_result"]["passed"] is True


def test_openapi_exposes_versioned_routes() -> None:
    paths = create_app().openapi()["paths"]
    assert "/api/v1/predictions/upload" in paths
    assert "/api/v1/explanations/activations" in paths


def test_raw_upload_contract_preserves_bytes() -> None:
    assert extract_image("image/png", b"image") == (b"image", "image/png")


def test_upload_rejects_unknown_content_type() -> None:
    try:
        extract_image("text/plain", b"not an image")
    except TypeError as error:
        assert "PNG or JPEG" in str(error)
    else:
        raise AssertionError("unsupported upload type was accepted")
