from app.main import app


def test_openapi_contains_required_versioned_routes() -> None:
    paths = set(app.openapi()["paths"])
    expected = {
        "/api/v1/health",
        "/api/v1/model-card",
        "/api/v1/models",
        "/api/v1/classes",
        "/api/v1/samples",
        "/api/v1/samples/{sample_id}",
        "/api/v1/predictions/sample",
        "/api/v1/predictions/compare",
        "/api/v1/evaluation/summary",
        "/api/v1/evaluation/{model_id}",
        "/api/v1/evaluation/errors",
        "/api/v1/traces/sample",
        "/api/v1/labs/cell-trace",
        "/api/v1/labs/gradient-flow",
    }
    assert expected <= paths
