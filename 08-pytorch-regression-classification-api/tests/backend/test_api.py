from __future__ import annotations

from fastapi.testclient import TestClient

from app.core.config import get_settings
from app.main import create_app


def client(bundle_root) -> TestClient:
    get_settings.cache_clear()
    settings = get_settings()
    settings.bundle_root = bundle_root
    return TestClient(create_app())


def test_liveness_is_independent_from_models(runtime_dir) -> None:
    with client(runtime_dir) as api:
        response = api.get("/api/v1/health/live")
        assert response.status_code == 200
        assert response.json() == {"status": "alive"}


def test_readiness_reports_missing_models(runtime_dir) -> None:
    with client(runtime_dir) as api:
        response = api.get("/api/v1/health/ready")
        assert response.json()["status"] == "degraded"


def test_unavailable_model_is_controlled(runtime_dir) -> None:
    with client(runtime_dir) as api:
        response = api.post("/api/v1/predict/regression", json={"features": {}})
        assert response.status_code == 503
