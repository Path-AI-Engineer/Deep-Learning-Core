from __future__ import annotations

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Query
from numpy.typing import NDArray

from app.core import LabRegistry, get_registry
from sequence_models.contracts import CHANNELS

router = APIRouter(prefix="/samples", tags=["samples"])


def _preview(values: NDArray[np.float32]) -> list[list[float]]:
    return [[round(float(value), 5) for value in row] for row in values[::16]]


@router.get("")
def samples(
    activity: str | None = None,
    subject: int | None = None,
    limit: int = Query(default=18, ge=1, le=24),
    registry: LabRegistry = Depends(get_registry),
) -> dict[str, object]:
    rows = registry.records
    if activity:
        rows = [row for row in rows if row.activity == activity.upper()]
    if subject is not None:
        rows = [row for row in rows if row.subject_id == subject]
    return {
        "items": [
            {
                "sample_id": row.sample_id,
                "activity": row.activity,
                "subject_id": f"subject-{row.subject_id}",
                "split": row.split,
                "sequence_length": 128,
                "channels": len(CHANNELS),
                "preview": _preview(row.values),
            }
            for row in rows[:limit]
        ],
        "count": min(len(rows), limit),
        "data_mode": registry.data_mode,
    }


@router.get("/{sample_id}")
def sample_detail(
    sample_id: str,
    registry: LabRegistry = Depends(get_registry),
) -> dict[str, object]:
    try:
        row = registry.require_sample(sample_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {
        "sample_id": row.sample_id,
        "activity": row.activity,
        "subject_id": f"subject-{row.subject_id}",
        "split": row.split,
        "sequence_length": 128,
        "channels": list(CHANNELS),
        "signals": row.values.round(6).T.tolist(),
        "data_mode": registry.data_mode,
    }
