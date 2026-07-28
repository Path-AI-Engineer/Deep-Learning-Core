from __future__ import annotations

from fastapi import APIRouter, Depends

from app.core import LabRegistry, get_registry
from app.schemas import CellTraceRequest

router = APIRouter(prefix="/labs", tags=["labs"])


@router.post("/cell-trace")
def educational_cell_trace(
    request: CellTraceRequest,
    registry: LabRegistry = Depends(get_registry),
) -> dict[str, object]:
    return registry.cell_trace(request.cell_type)


@router.get("/gradient-flow")
def gradient_flow(registry: LabRegistry = Depends(get_registry)) -> dict[str, object]:
    return registry.gradient_flow()
