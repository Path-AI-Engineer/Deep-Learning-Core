from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query

from app.core import LabRegistry, get_registry

router = APIRouter(prefix="/evaluation", tags=["evaluation"])


@router.get("/summary")
def summary(registry: LabRegistry = Depends(get_registry)) -> dict[str, object]:
    return registry.comparison


@router.get("/errors")
def errors(
    model_id: str,
    true_class: str | None = None,
    predicted_class: str | None = None,
    limit: int = Query(default=12, ge=1, le=24),
    registry: LabRegistry = Depends(get_registry),
) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    try:
        for record in registry.records:
            prediction = registry.predict(record.sample_id, model_id)
            if prediction["true_class"] == prediction["predicted_class"]:
                continue
            if true_class and prediction["true_class"] != true_class.upper():
                continue
            if predicted_class and prediction["predicted_class"] != predicted_class.upper():
                continue
            rows.append(prediction)
            if len(rows) >= limit:
                break
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {"items": rows, "count": len(rows)}


@router.get("/{model_id}")
def evaluation(
    model_id: str,
    registry: LabRegistry = Depends(get_registry),
) -> dict[str, object]:
    try:
        _, bundle = registry.require_model(model_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {
        "model_id": model_id,
        "version": bundle.version,
        "metrics": bundle.metrics,
        "manifest": {
            key: value
            for key, value in bundle.manifest.items()
            if key not in {"state_sha256"}
        },
        "warning": "Fixture evaluation is software evidence, not a UCI benchmark.",
    }
