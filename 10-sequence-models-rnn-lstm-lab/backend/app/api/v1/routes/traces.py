from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from app.core import LabRegistry, get_registry
from app.schemas import SampleTraceRequest

router = APIRouter(prefix="/traces", tags=["traces"])


@router.post("/sample")
def sample_trace(
    request: SampleTraceRequest,
    registry: LabRegistry = Depends(get_registry),
) -> dict[str, object]:
    if request.end_timestep < request.start_timestep:
        raise HTTPException(status_code=422, detail="end_timestep must be >= start_timestep")
    if request.end_timestep - request.start_timestep > 63:
        raise HTTPException(status_code=422, detail="a trace may expose at most 64 timesteps")
    try:
        return registry.sample_trace(
            request.sample_id,
            request.model_id,
            request.selected_units,
            request.start_timestep,
            request.end_timestep,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
