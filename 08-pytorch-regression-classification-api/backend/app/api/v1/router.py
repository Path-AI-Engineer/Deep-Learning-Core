from __future__ import annotations

from typing import Annotated
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, Request, status

from app.schemas import BatchPredictionRequest, PredictionRequest
from app.services import ModelRegistry

router = APIRouter(prefix="/api/v1")


def registry(request: Request) -> ModelRegistry:
    return request.app.state.registry


Registry = Annotated[ModelRegistry, Depends(registry)]


def _model(task: str, model_registry: ModelRegistry):
    if task not in {"regression", "classification"}:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Unknown task: {task}.",
        )
    try:
        return model_registry.get(task)
    except LookupError as error:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(error)) from error


@router.get("/health/live", tags=["health"])
def liveness() -> dict[str, str]:
    return {"status": "alive"}


@router.get("/health/ready", tags=["health"])
def readiness(model_registry: Registry) -> dict[str, object]:
    models = model_registry.readiness()
    return {"status": "ready" if all(models.values()) else "degraded", "models": models}


@router.get("/health", tags=["health"])
def health(model_registry: Registry) -> dict[str, object]:
    models = model_registry.readiness()
    return {
        "status": "ready" if all(models.values()) else "degraded",
        "api_version": "1.0.0",
        "models": models,
    }


@router.get("/tasks", tags=["catalog"])
def tasks(model_registry: Registry) -> dict[str, object]:
    return {"tasks": model_registry.tasks(), "batch_limit": 100}


@router.get("/tasks/{task}/schema", tags=["catalog"])
def schema(task: str, model_registry: Registry) -> dict[str, object]:
    predictor = _model(task, model_registry)
    return {
        "task": task,
        "dataset": predictor.metadata["dataset"],
        "feature_names": predictor.metadata["feature_names"],
        "features": predictor.metadata["feature_schema"],
        "class_names": predictor.metadata["class_names"],
        "target_unit": predictor.metadata["target_unit"],
        "examples": predictor.metadata["examples"],
    }


@router.get("/tasks/{task}/model-card", tags=["catalog"])
def model_card(task: str, model_registry: Registry) -> dict[str, object]:
    predictor = _model(task, model_registry)
    return {
        "task": task,
        "model_version": predictor.metadata["model_version"],
        "dataset": predictor.metadata["dataset"],
        "architecture": predictor.metadata["architecture"],
        "metrics": predictor.metadata["metrics"],
        "baseline_metrics": predictor.metadata["baseline_metrics"],
        "history": predictor.metadata["history"],
        "limitations": predictor.metadata["limitations"],
    }


@router.post("/predictions/{task}", tags=["inference"])
@router.post("/predict/{task}", tags=["inference"], include_in_schema=False)
def predict(
    task: str,
    payload: PredictionRequest,
    model_registry: Registry,
) -> dict[str, object]:
    predictor = _model(task, model_registry)
    try:
        result = predictor.predict([payload.features])[0]
    except ValueError as error:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(error)) from error
    return {
        "task": task,
        "model_version": predictor.metadata["model_version"],
        "prediction": result,
        "warnings": predictor.metadata["limitations"][:2],
        "request_id": str(uuid4()),
    }


@router.post("/predictions/{task}/batch", tags=["inference"])
@router.post("/predict/{task}/batch", tags=["inference"], include_in_schema=False)
def predict_batch(
    task: str,
    payload: BatchPredictionRequest,
    model_registry: Registry,
) -> dict[str, object]:
    if len(payload.rows) > 100:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="Batch inference accepts at most 100 observations.",
        )
    predictor = _model(task, model_registry)
    try:
        results = predictor.predict(payload.rows)
    except ValueError as error:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(error)) from error
    return {
        "task": task,
        "model_version": predictor.metadata["model_version"],
        "count": len(results),
        "predictions": results,
        "request_id": str(uuid4()),
    }
