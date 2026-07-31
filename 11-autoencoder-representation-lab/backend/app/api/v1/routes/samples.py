from __future__ import annotations

from fastapi import APIRouter, Depends, Query

from app.core import LabRegistry, get_registry

router = APIRouter(prefix="/samples", tags=["samples"])


@router.get("")
def samples(
    class_id: int | None = Query(default=None, ge=0, le=9),
    limit: int = Query(default=30, ge=1, le=48),
    registry: LabRegistry = Depends(get_registry),
) -> dict[str, object]:
    items = [
        item
        for item in registry.gallery
        if class_id is None or int(item["label"]) == class_id
    ][:limit]
    return {
        "items": items,
        "count": len(items),
        "data_mode": registry.data_mode,
    }


@router.get("/{sample_id}")
def sample_detail(
    sample_id: str,
    registry: LabRegistry = Depends(get_registry),
) -> dict[str, object]:
    image, label = registry.sample(sample_id)
    item = next(item for item in registry.gallery if item["sample_id"] == sample_id)
    return {
        **item,
        "shape": list(image.shape[1:]),
        "label": label,
        "available_reconstructions": [
            "mean-image",
            "pca",
            "dense-ae",
            "conv-ae",
            "denoising-ae",
            "latent-2d",
        ],
    }
