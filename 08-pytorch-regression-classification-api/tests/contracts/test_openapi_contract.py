from __future__ import annotations

from app.main import create_app


def test_openapi_contains_versioned_inference_routes() -> None:
    schema = create_app().openapi()
    assert schema["info"]["version"] == "1.0.0"
    assert "/api/v1/predictions/{task}" in schema["paths"]
    assert "/api/v1/predictions/{task}/batch" in schema["paths"]
    assert "/api/v1/health" in schema["paths"]
    assert "/api/v1/tasks/{task}/model-card" in schema["paths"]
