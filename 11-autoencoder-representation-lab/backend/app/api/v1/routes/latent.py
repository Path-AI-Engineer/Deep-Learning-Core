from __future__ import annotations

from fastapi import APIRouter, Depends, Query

from app.core import LabRegistry, get_registry
from app.schemas import InterpolateRequest, LatentDecodeRequest

router = APIRouter(prefix="/latent", tags=["latent"])


@router.get("/points")
def points(
    class_ids: str | None = None,
    limit: int = Query(default=200, ge=1, le=300),
    registry: LabRegistry = Depends(get_registry),
) -> dict[str, object]:
    selected: set[int] | None = None
    if class_ids:
        try:
            selected = {int(value) for value in class_ids.split(",")}
        except ValueError as error:
            raise ValueError("class_ids must be comma-separated integers") from error
        if not selected <= set(range(10)):
            raise ValueError("class_ids must be between 0 and 9")
    return registry.latent_points(selected, limit)


@router.get("/sample/{sample_id}")
@router.get("/samples/{sample_id}", include_in_schema=False)
def sample(
    sample_id: str,
    registry: LabRegistry = Depends(get_registry),
) -> dict[str, object]:
    return registry.latent_sample(sample_id)


@router.post("/decode")
def decode(
    request: LatentDecodeRequest,
    registry: LabRegistry = Depends(get_registry),
) -> dict[str, object]:
    return registry.decode(request.x, request.y)


@router.post("/interpolate")
def interpolate(
    request: InterpolateRequest,
    registry: LabRegistry = Depends(get_registry),
) -> dict[str, object]:
    return registry.interpolation(
        request.model_id,
        request.sample_id_a,
        request.sample_id_b,
        request.steps,
    )
