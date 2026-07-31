from __future__ import annotations

from fastapi import APIRouter, Depends, Query

from app.core import LabRegistry, get_registry

router = APIRouter(prefix="/evaluation", tags=["evaluation"])


@router.get("/summary")
def summary(registry: LabRegistry = Depends(get_registry)) -> dict[str, object]:
    return registry.comparison


@router.get("/model/{model_id}")
@router.get("/models/{model_id}", include_in_schema=False)
def model_evaluation(
    model_id: str,
    registry: LabRegistry = Depends(get_registry),
) -> dict[str, object]:
    registry.require_model(model_id)
    row = next(
        item for item in registry.model_rows() if item["model_id"] == model_id
    )
    return {
        **row,
        "error_examples": registry.error_rows(model_id, None, 5),
        "limitations": [
            "Validation evidence comes from the educational fixture.",
            "Reconstruction and probe metrics must be interpreted separately.",
        ],
    }


@router.get("/errors")
def errors(
    model_id: str,
    class_id: int | None = Query(default=None, ge=0, le=9),
    limit: int = Query(default=12, ge=1, le=30),
    registry: LabRegistry = Depends(get_registry),
) -> dict[str, object]:
    return {
        "model_id": model_id,
        "items": registry.error_rows(model_id, class_id, limit),
        "sort_by": "mse_desc",
    }
