from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from app.core import LabRegistry, get_registry
from app.schemas import CompareRequest, PredictionRequest

router = APIRouter(prefix="/predictions", tags=["predictions"])


@router.post("/sample")
def predict(
    request: PredictionRequest,
    registry: LabRegistry = Depends(get_registry),
) -> dict[str, object]:
    try:
        return registry.predict(request.sample_id, request.model_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/compare")
def compare(
    request: CompareRequest,
    registry: LabRegistry = Depends(get_registry),
) -> dict[str, object]:
    try:
        predictions = [
            registry.predict(request.sample_id, model_id) for model_id in request.model_ids
        ]
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    classes = [str(prediction["predicted_class"]) for prediction in predictions]
    return {
        "sample_id": request.sample_id,
        "predictions": predictions,
        "agreement": len(set(classes)) == 1,
        "disagreement": sorted(set(classes)) if len(set(classes)) > 1 else [],
        "warning": "Model agreement is not proof that the prediction is correct.",
    }
